import os
import sys

sys.path.append("..")

from loss import *
from model import *
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def Trainer(
    model,
    model_optimizer,
    classifier,
    classifier_optimizer,
    train_dl,
    valid_dl,
    test_dl,
    device,
    logger,
    configs,
    experiment_log_dir,
    training_mode,
):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")
    if training_mode == "pre_train":
        print("Pretraining on source dataset")
        for epoch in range(1, configs.num_epoch + 1):
            # Train and validate
            """Train. In fine-tuning, this part is also trained???"""
            train_loss = model_pretrain(
                model,
                model_optimizer,
                criterion,
                train_dl,
                configs,
                device,
                training_mode,
            )
            logger.debug(
                f"\nPre-training Epoch : {epoch}", f"Train Loss : {train_loss:.4f}"
            )

        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {"model_state_dict": model.state_dict()}
        torch.save(
            chkpoint, os.path.join(experiment_log_dir, "saved_models", f"ckp_last.pt")
        )
        print(
            "Pretrained model is stored at folder:{}".format(
                experiment_log_dir + "saved_models" + "ckp_last.pt"
            )
        )

    """Fine-tuning and Test"""
    if training_mode != "pre_train":
        """fine-tune"""
        print("Fine-tune on Fine-tuning set")
        performance_list = []
        total_f1 = []
        KNN_f1 = []
        global emb_finetune, label_finetune, emb_test, label_test

        for epoch in range(1, configs.num_epoch + 1):
            logger.debug(f"\nEpoch : {epoch}")

            valid_loss, emb_finetune, label_finetune, F1 = model_finetune(
                model,
                model_optimizer,
                valid_dl,
                configs,
                device,
                training_mode,
                classifier=classifier,
                classifier_optimizer=classifier_optimizer,
            )
            scheduler.step(valid_loss)

            # save best fine-tuning model""
            global arch
            arch = "sleepedf2eplipsy"
            if len(total_f1) == 0 or F1 > max(total_f1):
                print("update fine-tuned model")
                os.makedirs("experiments_logs/finetunemodel/", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    "experiments_logs/finetunemodel/" + arch + "_model.pt",
                )
                torch.save(
                    classifier.state_dict(),
                    "experiments_logs/finetunemodel/" + arch + "_classifier.pt",
                )
            total_f1.append(F1)

            # evaluate on the test set
            """Testing set"""
            logger.debug("Test on Target datasts test set")
            model.load_state_dict(
                torch.load("experiments_logs/finetunemodel/" + arch + "_model.pt")
            )
            classifier.load_state_dict(
                torch.load("experiments_logs/finetunemodel/" + arch + "_classifier.pt")
            )
            (
                test_loss,
                test_acc,
                test_auc,
                test_prc,
                emb_test,
                label_test,
                performance,
            ) = model_test(
                model,
                test_dl,
                configs,
                device,
                training_mode,
                classifier=classifier,
                classifier_optimizer=classifier_optimizer,
            )
            performance_list.append(performance)

            """Use KNN as another classifier; it's an alternation of the MLP classifier in function model_test. 
            Experiments show KNN and MLP may work differently in different settings, so here we provide both. """
            # train classifier: KNN
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(emb_finetune, label_finetune)
            knn_acc_train = neigh.score(emb_finetune, label_finetune)
            # print('KNN finetune acc:', knn_acc_train)
            representation_test = emb_test.detach().cpu().numpy()

            knn_result = neigh.predict(representation_test)
            knn_result_score = neigh.predict_proba(representation_test)
            one_hot_label_test = one_hot_encoding(label_test)
            # print(classification_report(label_test, knn_result, digits=4))
            # print(confusion_matrix(label_test, knn_result))
            knn_acc = accuracy_score(label_test, knn_result)
            precision = precision_score(
                label_test,
                knn_result,
                average="macro",
            )
            recall = recall_score(
                label_test,
                knn_result,
                average="macro",
            )
            F1 = f1_score(label_test, knn_result, average="macro")
            auc = roc_auc_score(
                one_hot_label_test, knn_result_score, average="macro", multi_class="ovr"
            )
            prc = average_precision_score(
                one_hot_label_test, knn_result_score, average="macro"
            )
            print(
                "KNN Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f"
                % (knn_acc, precision, recall, F1, auc, prc)
            )
            KNN_f1.append(F1)
        logger.debug(
            "\n################## Best testing performance! #########################"
        )
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:, 0], axis=0)]
        print(
            "Best Testing Performance: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f "
            "| AUPRC=%.4f"
            % (
                best_performance[0],
                best_performance[1],
                best_performance[2],
                best_performance[3],
                best_performance[4],
                best_performance[5],
            )
        )
        print("Best KNN F1", max(KNN_f1))

    logger.debug("\n################## Training is Done! #########################")

def generate_random_masks(mask_ratio, TSlength_aligned, batch_size):
    def single_sample_mask():
        idx = np.random.permutation(TSlength_aligned)[: int(mask_ratio * TSlength_aligned)]
        mask = np.zeros(TSlength_aligned)
        mask[idx] = 1
        return mask

    masks_list = [single_sample_mask() for _ in range(batch_size)]
    masks = np.stack(masks_list, axis=0)  # (num_samples, ts_size)
    return masks


def forward(self, x):
    # input is 2 dimension tensor [batch_size, z_dim, seq_len]
    batch_size, seq_len, z_dim = x.size()

    # generate random mask
    mask = self.generate_random_masks(batch_size)
    mask = torch.from_numpy(mask).to(dtype=torch.long, device=x.device)


def tokenize(x, token_size):
    """
    Tokenizes x into tokens of size token_size
    Assumes x is divisible by token_size
    """
    # x = (batch_size, feature_size, context_window)
    batch_size = x.shape[0]
    feature_size = x.shape[1]
    context_window = x.shape[2]
    num_tokens = int(context_window / token_size)
    x = x.reshape(batch_size, feature_size, num_tokens, token_size)
    # returns a tensor tokenized
    return x # (batch_size, features, num_tokens, token_size) 

def convert_mask_to_shape(rand_mask, token_size, feature_size):
    rand_mask_1 = np.expand_dims(rand_mask, axis=2)
    rand_mask_2 = np.repeat(rand_mask_1, token_size, axis=2)
    rand_mask_3 = np.expand_dims(rand_mask_2, axis=1)
    rand_mask_4 = np.repeat(rand_mask_3, feature_size, axis=1)
    return rand_mask_4

def generate_random_masks_mini_tokens(mask_ratio, num_tokens, batch_size):
    """ Generate random masks for the mini tokens """
    def single_sample_mask():
        idx = np.random.permutation(num_tokens)[: int(mask_ratio * num_tokens)]
        mask = np.zeros(num_tokens)
        mask[idx] = 1
        return mask
    masks_list = [single_sample_mask() for _ in range(batch_size)]
    masks = np.stack(masks_list, axis=0)  # (num_samples, num_tokens)
    return masks


def get_view_of_fixed_mask(x, mask, keep_tok, token_size, device):
    """ Return just the masked section """
    mask = torch.from_numpy(mask).to(device)
    
    idx = torch.argsort(mask, dim=1) # batch_size, keep_tok
    ids_restore = torch.argsort(idx, dim=1)
    idx = idx[:,:keep_tok]
    idx = torch.unsqueeze(idx, dim=1) # batch_size, feature_size, keep_tok
    idx = torch.unsqueeze(idx, dim=3) # batch_size, feature_size, keep_tok, token_size
    idx = idx.repeat(1, feature_size, 1, token_size)
    return torch.gather(x, dim=2, index=idx), ids_restore

def generate_random_masks(mask_ratio, TSlength_aligned, batch_size):
    def single_sample_mask():
        idx = np.random.permutation(TSlength_aligned)[: int(mask_ratio * TSlength_aligned)]
        mask = np.zeros(TSlength_aligned)
        mask[idx] = 1
        return mask

    masks_list = [single_sample_mask() for _ in range(batch_size)]
    masks = np.stack(masks_list, axis=0)  # (num_samples, ts_size)
    return masks

def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [batch_size, num_tokens, D], sequence
    """
    batch_size, feature_size, num_tokens, token_size = x.shape # (batch_size, feature_size, num_tokens, token_size)
    len_keep = int(num_tokens * (1 - mask_ratio))
    
    noise = torch.rand(batch_size, num_tokens, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, token_size))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, num_tokens], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

if __name__ == "__main__":
    pretrain_dataset = "SleepEEG"
    exec(f"from config_files.{pretrain_dataset}_Configs import Config as Configs")
    configs = Configs()
    with_gpu = torch.cuda.is_available()
    if with_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    data = torch.randn((128, 1, 180))
    aug1 = torch.randn((128, 1, 180))
    data_f = torch.randn((128, 1, 180))
    data = data.float().to(device)

    # data: [128, 1, 180], labels: [128]
    aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
    data_f = data_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

    feature_size = 1 # hardcode for now

    # (batch_size, feature_size, num_tokens, token_size)
    aug1_tokenized = tokenize(aug1, configs.mini_token_length)
    data_f_tokenized = tokenize(data_f, configs.mini_token_length)

    token_size = configs.mini_token_length
    """mask 40% of aug1 tokens"""
    num_tokens = int(configs.TSlength_aligned / configs.mini_token_length)
    aug1_mask_ratio = 0.4
    keep_tok_t = int((1 - aug1_mask_ratio) * num_tokens)
    # (batch_size, mask[num_tokens])
    mask_t_orig = generate_random_masks_mini_tokens(aug1_mask_ratio, num_tokens, configs.batch_size)
    mask_t = convert_mask_to_shape(mask_t_orig, configs.mini_token_length, feature_size)
    mask_t = torch.from_numpy(mask_t).to(dtype=torch.long, device=aug1.device)
    aug1_x = aug1_tokenized * mask_t
    aug1_x, ids_restore_x = get_view_of_fixed_mask(aug1_x, mask_t_orig, keep_tok_t, token_size, device)

    """mask 70% of data_f tokens"""
    data_f_mask_ratio = 0.7
    keep_tok_f = int((1 - data_f_mask_ratio) * num_tokens)
    mask_f_orig = generate_random_masks_mini_tokens(data_f_mask_ratio, num_tokens, configs.batch_size)
    mask_f = convert_mask_to_shape(mask_f_orig, configs.mini_token_length, feature_size)
    mask_f = torch.from_numpy(mask_f).to(dtype=torch.long, device=data_f.device)
    data_f_x = data_f_tokenized * mask_f # (batch_size, feature_size, num_tokens, token_size)
    data_f_x, ids_restore_f = get_view_of_fixed_mask(data_f_x, mask_f_orig, keep_tok_f, token_size, device)


    """Produce embeddings"""
    # z_t, z_f are encoded embeddings

    batch_size = aug1.shape[0]
    token_size = configs.mini_token_length

    t_enc_x = Time_Encoder(configs, aug1_mask_ratio).to(device)
    t_enc_f = Time_Encoder(configs, data_f_mask_ratio).to(device)

    print(f"PASSING")
    h_t, z_t = t_enc_x(aug1_x.reshape((batch_size, -1)))
    h_f, z_f = t_enc_f(data_f_x.reshape((batch_size, -1)))

    # unmasked_x = get_view_of_fixed_mask(aug1_tokenized, mask_t_orig, num_tokens - keep_tok_t, token_size, device, asc=False)
    # unmasked_f = get_view_of_fixed_mask(data_f_tokenized, mask_f_orig, num_tokens - keep_tok_f, token_size, device, asc=False)

    # """Concatenate the encoded embeddings with the masked data for both time and frequency domain to feed into the individual decoders"""
    
    # noise = torch.rand(batch_size, z_f.shape[1], device=device)  # noise in [0, 1]
        
    mask_tokens = torch.zeros(batch_size, z_f.shape[1], device=device)

    # sort noise for each sample
    # ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # ids_restore = torch.argsort(ids_shuffle, dim=1)
    

    z_t_full = torch.cat(z_t, mask_tokens, dim=1)
    z_t_full = torch.gather(z_t_full, dim=1, index=ids_restore_x.unsqueeze(-1).repeat(1, 1, z_t_full.shape[2]))  # unshuffle

    print(f"z_t_full: {z_t_full}")
    
    # for b in range(batch_size):

    #     # TODO: alter so that the masked ones go through as a batch

    #     aug1_list = []
    #     data_f_list = []
    #     for i in range(num_tokens):
    #         if mask_t_orig[b, i] == 1:
    #             # if mask use encode
    #             aug1_x_token = aug1_x[b, :, i, :].reshape(1, feature_size, token_size).to(device)
    #             print("aug_x_token", aug1_x_token.shape)
    #             h_t, z_t = t_enc_x(aug1_x_token)
    #             # aug1_list.append(z_t)
    #             aug1_list.append(z_t.cpu())
    #         # else:
    #         #     # else append orig token
    #         #     aug_1_orig_tok = aug1_tokenized[b, :, i, :].reshape(1, feature_size, token_size)
    #         #     aug1_list.append(aug_1_orig_tok.cpu())

    #         if mask_f_orig[b, i] == 1:
    #             # if mask use encode
    #             data_f_token = data_f_x[b, :, i, :].reshape(1, feature_size, token_size).to(device)
    #             h_f, z_f = t_enc_f(data_f_token)
    #             data_f_list.append(z_f.cpu())
    #         # else:
    #         #     # else append orig token
    #         #     data_f_orig_tok = data_f_tokenized[b, :, i, :].reshape(1, feature_size, token_size)
    #         #     data_f_list.append(data_f_orig_tok.cpu())

    #     # (num_tokens, num_features, )
    #     aug1_list = torch.cat(aug1_list, axis=0).to(device)
    #     data_f_list = torch.cat(data_f_list, axis=0).to(device)

    #     # token_list = [single_sample_mask() for _ in range(batch_size)]
    #     print(f"[viczhu] OK:", aug1_list.shape)
    #     print(f"[viczhu] OK2:", data_f_list.shape)
    #     break
    # h_t, z_t, h_f, z_f = model(data, data_f)
    # h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

    """Decoder reconstruction"""
    # z_t_x
    # z_f_x
    recon_t = Time_Decoder(z_t_x)
    recon_f = Freq_Decoder(z_f_x)

    """Compute Pre-train loss = Reconstruction loss on time domain + Reconstruction loss on frequency domain + L2 penalty between z_t and z_f + Barlow Twins loss"""

    """Compute Pre-train loss"""
    """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
    # nt_xent_criterion = NTXentLoss_poly(device, configs.batch_size, configs.Context_Cont.temperature,
    #                                configs.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

    # loss_t = nt_xent_criterion(h_t, h_t_aug)
    # loss_f = nt_xent_criterion(h_f, h_f_aug)
    # l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

    # l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
    # loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

    # lam = 0.2
    # loss = lam*(loss_t + loss_f) + l_TF

    # total_loss.append(loss.item())
    # loss.backward()
    # model_optimizer.step()