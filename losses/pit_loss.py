import torch

def permutation_invariant_training_loss(S_true,S_pred,num_speaker=2, only_real=False):
    """
    Permuatation invariant training
    Ref from here https://arxiv.org/pdf/1607.00325.pdf
    """
    def loss_pit_sample(T, P, b):
        k = 0
        loss_pairs = torch.zeros_like(torch.Tensor([0,0,0,0]))
        for i in range(num_speaker):
            for j in range(num_speaker):
                if only_real:
                    loss = torch.sum(torch.square(S_true[b,:,:,0,i]-S_pred[b,:,:,0,j]))
                else:
                    loss = torch.sum(torch.square(S_true[b,:,:,:,i]-S_pred[b,:,:,:,j]))
                loss_pairs[k] = loss
                k += 1

        per1 = loss_pairs[0] + loss_pairs[3]
        per2 = loss_pairs[1] + loss_pairs[2]

        if per1<per2:
            min_loss=per1
        else:
            min_loss=per2
        
        return min_loss



    loss_batch = torch.zeros(S_true.shape[0])

    for b in range(S_true.shape[0]):
        loss_batch[b] = loss_pit_sample(S_true, S_pred, b)
    
    loss = torch.mean(loss_batch) / (num_speaker*298*257*2)
    return loss

