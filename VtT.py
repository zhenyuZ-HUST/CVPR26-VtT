import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
import torch.nn as nn

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from Mamba_tool.mamba_blocks import VSSM
from Mamba_models.mamba_ffn_neck import MambaNeck

def get_grad_beta_updatae(beta, mean_sim_list, grad_steps):
    if(len(mean_sim_list) <= grad_steps):
        score = sum(mean_sim_list) / (len(mean_sim_list) - 1)
    else:
        score = sum(mean_sim_list[-grad_steps:]) / grad_steps
    if(score <=0 and len(mean_sim_list)):
        return -1
    else:
        return beta
class Mamba_Net(nn.Module):
    def __init__(self):
        super(Mamba_Net, self).__init__()
        self.mamba = MambaNeck(
                       version='ssm',#ss2d, ssm
                       in_channels=512,
                       out_channels=512,
                       feat_size=5,
                       num_layers=2,
                       use_residual_proj=True) # True

        self.ctx = nn.Sequential(
            #nn.Linear(512, 512 // reduction, bias=False),
            #nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, image_feat):
        ret = self.mamba(image_feat)
        ret = ret['out'] 
        return ret

_tokenizer = _Tokenizer()

def prograd_backward_and_update(model, optim, scaler, loss_a, loss_b, lambda_=1, names=None):
    # loss_b not increase is okay
    # loss_a has to decline
    optim.zero_grad()
    # get name of the model parameters

    # backward loss_a
    if not torch.isfinite(loss_b).all():
        raise FloatingPointError("Loss is infinite or NaN!")

    scaler.scale(loss_b).backward(retain_graph=True)
    # normalize gradient
    b_grads = []

    for name, p in model.named_parameters():
        if 'lora_' in name:
            b_grads.append(p.grad.clone())

    # optimizer don't step
    optim.zero_grad()

    # backward loss_a
    if not torch.isfinite(loss_a).all():
        raise FloatingPointError("Loss is infinite or NaN!")
    
    scaler.scale(loss_a).backward()
    i = 0

    mean_sim = torch.tensor(0).float().cuda()
    for name, p in model.named_parameters():
        if 'lora_' not in name:
            continue
        # calculate cosine distance
        b_grad = b_grads[i]
        b_grad_norm = b_grad / torch.linalg.norm(b_grad)
        a_grad = p.grad.clone()
        a_grad_norm = a_grad / torch.linalg.norm(a_grad)

        if(not torch.isnan(torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten())).any()):
            mean_sim += torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten())
        else:
            mean_sim += 1
        if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
            p.grad = a_grad - lambda_ * torch.dot(
                a_grad.flatten(), b_grad_norm.flatten()
            ) * b_grad_norm
        
        i+=1

    scaler.step(optim)

    return model, optim, scaler, (mean_sim / i).detach().cpu().numpy()

class Adapter(nn.Module):
    def __init__(self, clip_model, ctx_init, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.ctx = nn.Sequential(
            nn.Linear(768, 512 // reduction, bias=False),
            #nn.ReLU(inplace=True),
            nn.Linear(512 // reduction, 512, bias=False),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.ctx(x) 

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model_zs, logit_scale, test_loader):
    if args.dataset == "CropDisease":
        label_names=["Apple___Apple_scab",
                    "Apple___Black_rot",
                    "Apple___Cedar_apple_rust",
                    "Apple___healthy",
                    "Blueberry___healthy",
                    "Cherry_(including_sour)___Powdery_mildew",
                    "Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)___Common_rust_",
                    "Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy",
                    "Grape___Black_rot",
                    "Grape___Esca_(Black_Measles)",
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Grape___healthy",
                    "Orange___Haunglongbing_(Citrus_greening)",
                    "Peach___Bacterial_spot",
                    "Peach___healthy",
                    "Pepper,_bell___Bacterial_spot",
                    "Pepper,_bell___healthy",
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy",
                    "Raspberry___healthy",
                    "Soybean___healthy",
                    "Squash___Powdery_mildew",
                    "Strawberry___Leaf_scorch",
                    "Strawberry___healthy",
                    "Tomato___Bacterial_spot",
                    "Tomato___Early_blight",
                    "Tomato___Late_blight",
                    "Tomato___Leaf_Mold",
                    "Tomato___Septoria_leaf_spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy"]
        
    elif args.dataset == "EuroSAT":

        label_names=["Annual Crop Land",
                    "Forest",
                    "Herbaceous Vegetation Land",
                    "Highway or Road",
                    "Industrial Buildings",
                    "Pasture Land",
                    "Permanent Crop Land",
                    "Residential Buildings",
                    "River",
                    "Sea or Lake",]
    elif args.dataset == "ISIC":

        label_names=["Melanoma",
                     "Melanocytic Nevus",
                     "Basal Cell Carcinoma",
                     "Actinic Keratosis",
                     "Benign Keratosis",
                     "Dermatofibroma",
                     "Vascular Lesion"]

    elif args.dataset == "ChestX":
        label_names=["Atelectasis",
                     "Cardiomegaly",
                     "Effusion",
                     "Infiltration",
                     "Mass",
                     "Nodule",
                     "Pneumothorax"]
    import copy
    import numpy as np
    VALIDATION = False
    total_iters = args.epochs
    zs_acc_list = []
    fine_acc_list = []
    fine40_acc_list = []
    fine80_acc_list = []
    fine120_acc_list = []
    fine160_acc_list = []

    combine_acc_list = []
    mae_acc_list = []
    both_mae_acc_list = []


    list_lora_layers = apply_lora(args, clip_model_zs)
    clip_model_zs = clip_model_zs.cuda() 
    for idx, (t_all, x_all, y_all) in enumerate(test_loader):
        mean_sim_list = [1]
        Used_beta = args.beta ###

        clip_model = copy.deepcopy(clip_model_zs)
        mamba_net= Mamba_Net()
        mamba_net = mamba_net.cuda()
        mamba_params = []
        for name, param in mamba_net.named_parameters():
            mamba_params.append(param)

        lora_parameters = get_lora_parameters(clip_model)

        aug_num = len(x_all)
        test_data = x_all[0].cuda()
        test_label = torch.tensor(np.repeat(range(args.way), args.shot + 15)).cuda()

        all_image = torch.stack(x_all, dim=0).cuda()
        all_labels = torch.tensor(np.repeat(range(args.way), args.shot + 15)).cuda().reshape(args.way, args.shot + 15)
        all_labels = torch.unsqueeze(all_labels, dim=0).repeat(aug_num,1,1) 

        supp_images = all_image[:, :, :args.shot,:,:,:].reshape(args.way * args.shot * aug_num, 3,224,224)

        supp_images_noAug = test_data[:, :args.shot,:,:,:].reshape(args.way * args.shot, 3,224,224)
        query_images = test_data[:, args.shot:,:,:,:].reshape(args.way * 15, 3,224,224)

        supp_label = all_labels[:,:,:args.shot].reshape(-1)

        supp_label_noAug = test_label.reshape(args.way, args.shot + 15)[:,:args.shot].reshape(-1)
        query_label = test_label.reshape(args.way, args.shot + 15)[:,args.shot:].reshape(-1)

        labels_list = y_all[0][:,0]

        class_texts = [label_names[i] for i in labels_list]


        #################################
        #supp_images = supp_images_noAug
        #supp_label = supp_label_noAug
        ###################################################################
        zs_acc = 0#fsl_test(clip_model, query_images, query_label, class_texts)
        ####################################################################
        with torch.no_grad(): 
            template = 'a photo of a {}.'#dataset.template[0] 
            full_texts = [template.format(classname.replace('_', ' ')) for classname in class_texts]
            emp_full_texts = [template.format(''.replace('_', ' ')) for classname in class_texts]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(full_texts).cuda()
                class_embeddings, class_embeddings_all = clip_model.encode_text(texts, ret_all = True)
                emp_texts = clip.tokenize(emp_full_texts).cuda()
                emp_class_embeddings, emp_class_embeddings_all = clip_model.encode_text(emp_texts, ret_all = True)
            text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
            
        clip_model.train()
        batch_size = 25
        support_size = supp_images.size(0)


        ########################################################################################### unsupervised L
        total_iters = int(total_iters)
        parameters_to_update = [{'params': mamba_params,'lr':args.mamba_lr}, {'params': lora_parameters}]
        ##############################################
        optimizer = torch.optim.AdamW(parameters_to_update, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters * 8, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler()

        count_iters = 0
        while count_iters < total_iters:
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = supp_images[selected_id]
                y_batch = supp_label[selected_id] 
                if(args.encoder == 'both' or args.encoder == 'text'):
                    template = 'a photo of a {}.'#dataset.template[0] 
                    full_texts = [template.format(classname.replace('_', ' ')) for classname in class_texts]
                    emp_full_texts = [template.format(''.replace('_', ' ')) for classname in class_texts]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        texts = clip.tokenize(full_texts).cuda()
                        class_embeddings, class_embeddings_all = clip_model.encode_text(texts, ret_all = True)
                        emp_texts = clip.tokenize(emp_full_texts).cuda()
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features_raw, image_features_raw_all = clip_model.encode_image(z_batch, ret_all = True)
                    supp_image_features_raw, supp_image_features_raw_all = clip_model.encode_image(supp_images_noAug, ret_all = True)
                image_features = image_features_raw/image_features_raw.norm(dim=-1, keepdim=True)
                supp_image_features = supp_image_features_raw/supp_image_features_raw.norm(dim=-1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                ce_loss = F.cross_entropy(cosine_similarity, y_batch)
                

                ################### cross scan
                batch_class_embeddings_all = class_embeddings_all[y_batch]
                cat_input = [torch.unsqueeze(image_features_raw, dim=1)]
                #cat_input = []
                for i in range(image_features_raw_all.size(1), -1, -1):
                    i-=1
                    # if(i == 11):
                    #     continue
                    if(i == -1):
                        break
                    
                    test = image_features_raw_all.size(1) - i - 1
                    v_patch = image_features_raw_all[:,i]
                    t_patch = batch_class_embeddings_all[:,i]
                    cat_input.append(torch.unsqueeze(t_patch, dim=1))
                    cat_input.append(torch.unsqueeze(v_patch, dim=1))

                cat_input = torch.cat(cat_input, dim=1)
                cat_input = cat_input.view(cat_input.size(0), 5,5,cat_input.size(-1))
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_mae_encode = mamba_net(cat_input)
                    

                ######################################### absorb
                template = 'a photo of a x.'#dataset.template[0] 
                temp_texts = [template for i in range(len(image_mae_encode))]
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(temp_texts).cuda()
                    mae_image_embeddings = clip_model.encode_text(texts, image_mae_encode)
                mae_text_features = mae_image_embeddings/mae_image_embeddings.norm(dim=-1, keepdim=True)
                
    

                mae_cosine_similarity = image_features @ mae_text_features.t()
                mae_loss = - torch.diag(mae_cosine_similarity).mean()#F.cross_entropy(mae_cosine_similarity, y_batch)

                ##################### grad cut
                if(Used_beta > 0):
                    loss = ce_loss +  Used_beta * mae_loss
                    clip_model, optimizer, scaler, mean_sim = prograd_backward_and_update(clip_model, optimizer, scaler, loss, ce_loss, 1.0)
                    mean_sim_list.append(mean_sim)
                    Used_beta = get_grad_beta_updatae(args.beta, mean_sim_list, int(args.grad_steps))
                else:
                    loss = ce_loss
                    optimizer.zero_grad()
                    scaler.scale(ce_loss).backward()
                    scaler.step(optimizer)
                
                    
                scaler.update()
                scheduler.step()
                count_iters+=1
                save_freq = int(total_iters / 5)
                if count_iters == 1:
                    #acc40, _, _ , _= 0#fsl_test(clip_model, query_images, query_label, full_texts, mamba_net, supp_images_noAug, supp_label_noAug, emp_full_texts, give_text = text_features)
                    acc40 = 0
                    fine40_acc_list.append(acc40)
                    clip_model.train()
                if count_iters == 2 * save_freq:
                    #acc80, _, _, _ = 0#fsl_test(clip_model, query_images, query_label, full_texts, mamba_net, supp_images_noAug, supp_label_noAug, emp_full_texts, give_text = text_features)
                    acc80 = 0
                    fine80_acc_list.append(acc80)
                    clip_model.train()
                if count_iters == 3 * save_freq:
                    #acc120, _, _, _ = 0#fsl_test(clip_model, query_images, query_label, full_texts, mamba_net, supp_images_noAug, supp_label_noAug, emp_full_texts, give_text = text_features)
                    acc120 = 0
                    fine120_acc_list.append(acc120)
                    clip_model.train()
                if count_iters == 4 * save_freq:
                    #acc160, _, _, _ = 0#fsl_test(clip_model, query_images, query_label, full_texts, mamba_net, supp_images_noAug, supp_label_noAug, emp_full_texts, give_text = text_features)
                    acc160 = 0
                    fine160_acc_list.append(acc160)
                    clip_model.train()
                if count_iters == total_iters:
                    break

        ###################################################################
        fine_acc, acc_conbine, acc_mae, acc_both_mae = fsl_test(clip_model, query_images, query_label, full_texts, mamba_net, supp_images_noAug, supp_label_noAug, emp_full_texts, give_text = text_features)

        zs_acc_list.append(zs_acc)
        fine_acc_list.append(fine_acc)
        combine_acc_list.append(acc_conbine)
        mae_acc_list.append(acc_mae)
        both_mae_acc_list.append(acc_both_mae)

        if(idx % 1 == 0):
            #print("%d episods: zero shot acc is %g ,finetune acc is %g (ind.)" % (idx, np.mean(np.array(zs_acc_list)), np.mean(np.array(fine_acc_list))))
            #if(args.shot == 1):
            print("%d episods: zero shot acc is %g || 1/5 acc is %g , 2/5 acc is %g , 3/5 acc is %g, 4/5 acc is %g, full acc is %g | %g | %g | %g." % (idx, np.mean(np.array(zs_acc_list)), np.mean(np.array(fine40_acc_list)), np.mean(np.array(fine80_acc_list)), np.mean(np.array(fine120_acc_list)), np.mean(np.array(fine160_acc_list)), np.mean(np.array(fine_acc_list)), np.mean(np.array(combine_acc_list)), np.mean(np.array(mae_acc_list)), np.mean(np.array(both_mae_acc_list))))
            #else:
            #    print("%d episods: zero shot acc is %g || 200E acc is %g , 400E acc is %g , 600E acc is %g, 800E acc is %g, 1000E acc is %g." % (idx, np.mean(np.array(zs_acc_list)), np.mean(np.array(fine40_acc_list)), np.mean(np.array(fine80_acc_list)), np.mean(np.array(fine120_acc_list)), np.mean(np.array(fine160_acc_list)), np.mean(np.array(fine_acc_list))))

    
def fsl_test(clip_model, query_images, query_label, class_texts, mamba_net, supp_image, supp_label_noAug, emp_full_texts, give_text=None):
    clip_model.eval()
    with torch.no_grad(): 
        texts = class_texts
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            emp_full_texts = clip.tokenize(emp_full_texts).cuda()
            class_embeddings, class_embeddings_all = clip_model.encode_text(texts, ret_all=True)
            emp_class_embeddings, emp_class_embeddings_all = clip_model.encode_text(emp_full_texts, ret_all=True)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)


    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        images, target = query_images.cuda(), query_label.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features, image_features_raw_all = clip_model.encode_image(images, ret_all=True)
            supp_image_features, supp_image_features_raw_all = clip_model.encode_image(supp_image, ret_all=True)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = image_features @ text_features.t()
        acc = cls_acc(cosine_similarity, target) * len(cosine_similarity)
        tot_samples = len(cosine_similarity)
        acc /= tot_samples


        class_embeddings_all = class_embeddings_all[supp_label_noAug]
        cat_input = [torch.unsqueeze(supp_image_features, dim=1)]
        #cat_input = []
        for i in range(supp_image_features_raw_all.size(1), -1, -1):
            i-=1
            # if(i == 11):
            #     continue            
            if(i == -1):
                break
            v_patch = supp_image_features_raw_all[:,i]
            t_patch = class_embeddings_all[:,i]
            cat_input.append(torch.unsqueeze(t_patch, dim=1))
            cat_input.append(torch.unsqueeze(v_patch, dim=1))

        cat_input = torch.cat(cat_input, dim=1)
        cat_input = cat_input.view(cat_input.size(0), 5,5,cat_input.size(-1))
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_mae_encode = mamba_net(cat_input)
            
        #################################################################
        template = 'a photo of a x.'#dataset.template[0] 
        temp_texts = [template for i in range(len(image_mae_encode))]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(temp_texts).cuda()
            mae_image_embeddings = clip_model.encode_text(texts, image_mae_encode)
        if(mae_image_embeddings.size(0) != 5):
            mae_image_embeddings = mae_image_embeddings.view(5,-1,mae_image_embeddings.size(-1)).mean(1)
        mae_supp_text_features = mae_image_embeddings/mae_image_embeddings.norm(dim=-1, keepdim=True)
        

        mae_supp_cosine_similarity = image_features @ mae_supp_text_features.t()
        mae_comb_cosine_similarity = cosine_similarity * 0.5 + mae_supp_cosine_similarity * 0.5#mae_supp_cosine_similarity

        mae_supp_acc = cls_acc(mae_supp_cosine_similarity, target) #* len(mae_supp_cosine_similarity)
        mae_comb_acc = cls_acc(mae_comb_cosine_similarity, target) #* len(mae_both_cosine_similarity)
        #########################
        new_class_embediing = class_embeddings_all[:,5]
        new_text_features = new_class_embediing/new_class_embediing.norm(dim=-1, keepdim=True)
        cali_cosine_similarity = image_features @ new_text_features.t()
        cali_acc = cls_acc(cali_cosine_similarity, target) * len(cali_cosine_similarity)
        tot_samples = len(cosine_similarity)
        cali_acc /= tot_samples


    return acc, mae_supp_acc, cali_acc, mae_comb_acc





    