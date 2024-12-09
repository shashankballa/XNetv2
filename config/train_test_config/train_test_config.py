import numpy as np
from config.eval_config.eval import evaluate, evaluate_multi
import torch
import os
from PIL import Image
import torchio as tio
import copy


def print_train_loss_XNetv2(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus, batch_size = None):
    if batch_size is None:
        batch_size = num_batches['train_sup']
    train_epoch_loss_sup1 = train_loss_sup_1 / batch_size
    train_epoch_loss_sup2 = train_loss_sup_2 / batch_size
    train_epoch_loss_sup3 = train_loss_sup_3 / batch_size
    train_epoch_loss_unsup = train_loss_unsup / batch_size
    train_epoch_loss = train_loss / batch_size
    print('-' * print_num)
    print('| Train Sup Loss 1: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Train Sup Loss 2: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_minus, ' '), '|')
    print('| Train Sup Loss 3: {:.4f}'.format(train_epoch_loss_sup3).ljust(print_num_minus, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_unsup).ljust(print_num_minus, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss

def print_val_loss_XNetv2(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_minus):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches['val']
    val_epoch_loss_sup3 = val_loss_sup_3 / num_batches['val']
    print('-' * print_num)
    print('| Val Sup Loss 1: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Val Sup Loss 2: {:.4f}'.format(val_epoch_loss_sup2).ljust(print_num_minus, ' '), '|')
    print('| Val Sup Loss 3: {:.4f}'.format(val_epoch_loss_sup3).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3

def print_train_loss_WaveNetX(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss, num_batches, print_num, print_num_minus, batch_size = None):
    if batch_size is None:
        batch_size = num_batches['train_sup']
    train_epoch_loss_sup1 = train_loss_sup_1 / batch_size
    train_epoch_loss_sup2 = train_loss_sup_2 / batch_size
    train_epoch_loss_sup3 = train_loss_sup_3 / batch_size
    train_epoch_loss = train_loss / batch_size
    print('-' * print_num)
    print('| Train Sup Loss 1: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Train Sup Loss 2: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_minus, ' '), '|')
    print('| Train Sup Loss 3: {:.4f}'.format(train_epoch_loss_sup3).ljust(print_num_minus, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss

def print_val_loss_WaveNetX(val_loss_sup_1, num_batches, print_num, print_num_minus):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    print('-' * print_num)
    print('| Val Sup Loss 1: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1


def print_train_eval_sup(num_classes, score_list_train, mask_list_train, print_num):

    if num_classes == 2:
        eval_list = evaluate(score_list_train, mask_list_train)
        print('| Train Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]
        train_m_dc = eval_list[2]

    else:
        eval_list = evaluate_multi(score_list_train, mask_list_train)

        np.set_printoptions(precision=4, suppress=True)
        print('| Train  Jc: {}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Train mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[0]
        train_m_dc = eval_list[1]

    return eval_list, train_m_jc, train_m_dc

def print_val_eval_sup(num_classes, score_list_val, mask_list_val, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_val, mask_list_val)
        print('| Val Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
        val_m_dice = eval_list[2]
    else:
        eval_list = evaluate_multi(score_list_val, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Val mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[0]
        val_m_dice = eval_list[2]
    return eval_list, val_m_jc, val_m_dice

def save_val_best_sup_2d(num_classes, best_list, model, score_list_val, name_list_val, eval_list, path_trained_model, path_seg_results, palette, model_name):
    def save_results(pred_results, name_list, palette, save_path):
        assert len(name_list) == pred_results.shape[0], "Mismatch in results and name list lengths."
        for i in range(len(name_list)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(save_path, name_list[i]))

    if best_list[1] < eval_list[1]:
        best_list[1] = eval_list[1]  # Update in place

        torch.save(model.state_dict(), os.path.join(path_trained_model, f'Jc_{best_list[1]:.6f}_{model_name}.pth'))

        if num_classes == 2:
            score_list_val = torch.softmax(score_list_val, dim=1)
            pred_results = score_list_val[:, 1, :, :].cpu().numpy()
            pred_results = (pred_results > eval_list[0]).astype(np.uint8)  # Thresholding for binary
        else:
            pred_results = torch.max(score_list_val, 1)[1].cpu().numpy()  # Multiclass

        save_results(pred_results, name_list_val, palette, path_seg_results)
    elif best_list[2] < eval_list[2]:
        best_list[2] = eval_list[2]
        torch.save(model.state_dict(), os.path.join(path_trained_model, f'Dc_{best_list[2]:.6f}_{model_name}.pth'))
        
        if num_classes == 2:
            score_list_val = torch.softmax(score_list_val, dim=1)
            pred_results = score_list_val[:, 1, :, :].cpu().numpy()
            pred_results = (pred_results > eval_list[0]).astype(np.uint8)
        else:
            pred_results = torch.max(score_list_val, 1)[1].cpu().numpy()

        save_results(pred_results, name_list_val, palette, path_seg_results)

    return best_list

def save_results(pred_results, name_list, palette, save_path):
    """
    Save predicted segmentation maps with color palettes applied.
    """
    assert len(name_list) == pred_results.shape[0], "Mismatch in number of results and names."
    for i in range(len(name_list)):
        # Create a paletted image
        color_image = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
        color_image.putpalette(palette)
        color_image.save(os.path.join(save_path, name_list[i]))

def save_val_sup_2d_best_model(num_classes, best_list, model, best_model, score_list_val, name_list_val, eval_list, 
                               path_trained_model, path_seg_results, palette, model_name):
    """
    Save the best model based on the evaluation metrics and also save the segmentation predictions.

    Parameters:
    -----------
    num_classes : int
        Number of classes in the segmentation task.
    best_list : list
        A list containing the current best metrics. Typically best_list = [best_threshold, best_Jc, best_Dc, ...].
    model : torch.nn.Module
        The current model being evaluated.
    best_model : torch.nn.Module
        The currently stored best model.
    score_list_val : torch.Tensor
        The model output scores for the validation set (e.g., shape: [N, C, H, W]).
    name_list_val : list of str
        The list of image names corresponding to `score_list_val`.
    eval_list : list
        A list containing the new evaluation metrics (e.g., [threshold, Jaccard, Dice]).
    path_trained_model : str
        Directory path to save the model checkpoints.
    path_seg_results : str
        Directory path to save the segmentation results.
    palette : list
        A list representing the color palette for segmentation visualization.
    model_name : str
        A unique name/identifier for the model.
    """

    # Compare the current metrics (eval_list) with the stored best metrics (best_list)
    # Typically:
    #   best_list[0]: best threshold
    #   best_list[1]: best Jaccard Coefficient (Jc)
    #   best_list[2]: best Dice Coefficient (Dc)
    #   eval_list[0]: current threshold
    #   eval_list[1]: current Jaccard Coefficient
    #   eval_list[2]: current Dice Coefficient

    # Check if the new Jaccard Coefficient is better
    if best_list[1] < eval_list[1]:
        # Update best Jaccard
        best_list[1] = eval_list[1]
        best_model.load_state_dict(copy.deepcopy(model.state_dict()))

        # Save the best model based on Jaccard
        torch.save(model.state_dict(), os.path.join(path_trained_model, f'Jc_{best_list[1]:.6f}_{model_name}.pth'))

        # Compute predictions
        if num_classes == 2:
            # For binary segmentation, apply softmax and threshold
            score_list_val = torch.softmax(score_list_val, dim=1)
            pred_results = score_list_val[:, 1, :, :].cpu().numpy()
            pred_results = (pred_results > eval_list[0]).astype(np.uint8)
        else:
            # For multi-class segmentation, take the argmax
            pred_results = torch.argmax(score_list_val, dim=1).cpu().numpy()

        # Save the segmentation predictions
        save_results(pred_results, name_list_val, palette, path_seg_results)

    # If not improved on Jaccard, check Dice
    elif best_list[2] < eval_list[2]:
        # Update best Dice
        best_list[2] = eval_list[2]
        # best_model.load_state_dict(copy.deepcopy(model.state_dict()))

        # Save the best model based on Dice
        torch.save(model.state_dict(), os.path.join(path_trained_model, f'Dc_{best_list[2]:.6f}_{model_name}.pth'))

        # Compute predictions
        if num_classes == 2:
            score_list_val = torch.softmax(score_list_val, dim=1)
            pred_results = score_list_val[:, 1, :, :].cpu().numpy()
            pred_results = (pred_results > eval_list[0]).astype(np.uint8)
        else:
            pred_results = torch.argmax(score_list_val, dim=1).cpu().numpy()

        # Save segmentation predictions
        save_results(pred_results, name_list_val, palette, path_seg_results)

    return best_list, best_model

def save_val_best_sup_3d(best_list, model, eval_list, path_trained_model, model_name):

    if best_list[1] < eval_list[1]:
        best_list = eval_list
    torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, eval_list[1])))
    return best_list


def draw_pred_sup(num_classes, mask_train_sup, mask_val, pred_train_sup, outputs_val, train_eval_list, val_eval_list):

    mask_image_train_sup = mask_train_sup[0, :, :].data.cpu().numpy()
    mask_image_val = mask_val[0, :, :].data.cpu().numpy()

    if num_classes == 2:
        pred_image_train_sup = pred_train_sup[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup[pred_image_train_sup > train_eval_list[0]] = 1
        pred_image_train_sup[pred_image_train_sup <= train_eval_list[0]] = 0

        pred_image_val = outputs_val[0, 1, :, :].data.cpu().numpy()
        pred_image_val[pred_image_val > val_eval_list[0]] = 1
        pred_image_val[pred_image_val <= val_eval_list[0]] = 0

    else:
        pred_image_train_sup = torch.max(pred_train_sup, 1)[1]
        pred_image_train_sup = pred_image_train_sup[0, :, :].cpu().numpy()

        pred_image_val = torch.max(outputs_val, 1)[1]
        pred_image_val = pred_image_val[0, :, :].cpu().numpy()

    return mask_image_train_sup, pred_image_train_sup, mask_image_val, pred_image_val


def print_best_sup(num_classes, best_val_list, print_num):
    if num_classes == 2:
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:
        np.set_printoptions(precision=4, suppress=True)
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')


def print_test_eval(num_classes, score_list_test, mask_list_test, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_test, mask_list_test)
        print('| Test Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
    else:
        eval_list = evaluate_multi(score_list_test, mask_list_test)
        np.set_printoptions(precision=4, suppress=True)
        print('| Test  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Test mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')

    return eval_list


def save_test_2d(num_classes, score_list_test, name_list_test, threshold, path_seg_results, palette):

    if num_classes == 2:
        score_list_test = torch.softmax(score_list_test, dim=1)
        pred_results = score_list_test[:, 1, ...].cpu().numpy()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

    else:
        pred_results = torch.max(score_list_test, 1)[1]
        pred_results = pred_results.cpu().numpy()

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

def save_test_3d(num_classes, score_test, name_test, threshold, path_seg_results, affine):

    if num_classes == 2:
        score_list_test = torch.softmax(score_test, dim=0)
        pred_results = score_list_test[1, ...].cpu()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))

    else:
        pred_results = torch.max(score_test, 0)[1]
        pred_results = pred_results.cpu()
        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))


def print_train_loss_WaveNetX4(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss, num_batches, print_num, print_num_minus, batch_size = None):
    if batch_size is None:
        batch_size = num_batches['train_sup']
    train_epoch_loss_sup1 = train_loss_sup_1 / batch_size
    train_epoch_loss_sup2 = train_loss_sup_2 / batch_size
    train_epoch_loss_sup3 = train_loss_sup_3 / batch_size
    train_epoch_loss = train_loss / batch_size
    print('-' * print_num)
    print('| Train Super Loss: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Train fb_l0 Loss: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_minus, ' '), '|')
    print('| Train fb_l1 Loss: {:.4f}'.format(train_epoch_loss_sup3).ljust(print_num_minus, ' '), '|')
    print('| Total train Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss

def print_val_loss_WaveNetX4(val_loss_sup_1, val_loss_sup_best, num_batches, print_num, print_num_minus):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup_best = val_loss_sup_best / num_batches['val']
    print('-' * print_num)
    print('| Val  Super Loss : {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Best Super Loss : {:.4f}'.format(val_epoch_loss_sup_best).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup_best

def print_val_scores_WaveNetX4(num_classes, score_list_val, score_list_val_best, mask_list_val, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_val, mask_list_val)
        eval_list_best = evaluate(score_list_val_best, mask_list_val)
        print('| Val  Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val   Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val   Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Best Thr: {:.4f}'.format(eval_list_best[0]).ljust(print_num, ' '), '|')
        print('| Best  Jc: {:.4f}'.format(eval_list_best[1]).ljust(print_num, ' '), '|')
        print('| Best  Dc: {:.4f}'.format(eval_list_best[2]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
        val_m_dice = eval_list[2]
        best_jc = eval_list_best[1]
        best_dice = eval_list_best[2]
    else:
        eval_list = evaluate_multi(score_list_val, mask_list_val)
        eval_list_best = evaluate_multi(score_list_val_best, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Val mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[0]
        val_m_dice = eval_list[2]
        best_jc = eval_list_best[0]
        best_dice = eval_list_best[2]
    return eval_list, eval_list_best, val_m_jc, val_m_dice, best_jc, best_dice

def draw_pred_sup_WaveNetX4(num_classes, mask_train_sup, mask_val, pred_train_sup, pred_val, pred_val_best, train_eval_list, val_eval_list, val_eval_list_best):

    mask_image_train_sup = mask_train_sup[0, :, :].data.cpu().numpy()
    mask_image_val = mask_val[0, :, :].data.cpu().numpy()

    if num_classes == 2:
        pred_image_train_sup = pred_train_sup[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup[pred_image_train_sup > train_eval_list[0]] = 1
        pred_image_train_sup[pred_image_train_sup <= train_eval_list[0]] = 0

        pred_image_val = pred_val[0, 1, :, :].data.cpu().numpy()
        pred_image_val[pred_image_val > val_eval_list[0]] = 1
        pred_image_val[pred_image_val <= val_eval_list[0]] = 0

        pred_image_val_best = pred_val_best[0, 1, :, :].data.cpu().numpy()
        pred_image_val_best[pred_image_val_best > val_eval_list_best[0]] = 1
        pred_image_val_best[pred_image_val_best <= val_eval_list_best[0]] = 0

    else:
        pred_image_train_sup = torch.max(pred_train_sup, 1)[1]
        pred_image_train_sup = pred_image_train_sup[0, :, :].cpu().numpy()

        pred_image_val = torch.max(pred_val, 1)[1]
        pred_image_val = pred_image_val[0, :, :].cpu().numpy()

        pred_image_val_best = torch.max(pred_val_best, 1)[1]
        pred_image_val_best = pred_image_val_best[0, :, :].cpu().numpy()

    return mask_image_train_sup, pred_image_train_sup, mask_image_val, pred_image_val, pred_image_val_best