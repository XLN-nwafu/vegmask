import math
import torch


def diou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # Eqn. (7)
    loss = 1 - iou + (distance / diag_len)
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def ciou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # width and height of boxes
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    v = (4 / (math.pi**2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # Eqn. (10)
    loss = 1 - iou + (distance / diag_len) + alpha * v
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta_RS=0.50, eps=1e-10):

        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta_RS
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))

        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error = torch.zeros(fg_num).cuda()
        ranking_error = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations = fg_logits - fg_logits[ii]
            bg_relations = relevant_bg_logits - fg_logits[ii]

            if delta_RS > 0:
                fg_relations = torch.clamp(fg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
                bg_relations = torch.clamp(bg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos = torch.sum(fg_relations)
            FP_num = torch.sum(bg_relations)

            # Rank of ii among all examples
            rank = rank_pos + FP_num

            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii] = FP_num / rank

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations * (1 - fg_targets)) / rank_pos

            # Find examples in the target sorted order for example ii
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            # The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            # Compute target sorting error. (Eq. 8)
            # Since target ranking error is 0, this is also total target error
            target_sorting_error = torch.sum(target_sorted_order * (1 - fg_targets)) / rank_pos_target

            # Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error

            # Identity Update for Ranking Error
            if FP_num > eps:
                # For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                # For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations * (ranking_error[ii] / FP_num))

            # Find the positives that are misranked (the cause of the error)
            # These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            # Denominotor of sorting pmf
            sorting_pmf_denom = torch.sum(missorted_examples)

            # Identity Update for Sorting Error
            if sorting_pmf_denom > eps:
                # For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                # For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples * (sorting_error[ii] / sorting_pmf_denom))

        # Normalize gradients by number of positives
        classification_grads[fg_labels] = (fg_grad / fg_num)
        classification_grads[relevant_bg_labels] = (relevant_bg_grad / fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None, None


class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5):
        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta

        # Get valid bg logits
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))
        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        rank = torch.zeros(fg_num).cuda()
        prec = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        max_prec = 0
        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # x_ij s as score differences with fgs
            fg_relations = fg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with fgs
            fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)
            # Discard i=j in the summation in rank_pos
            fg_relations[ii] = 0

            # x_ij s as score differences with bgs
            bg_relations = relevant_bg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with bgs
            bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)

            # Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos = 1 + torch.sum(fg_relations)
            FP_num = torch.sum(bg_relations)
            # Store the total since it is normalizer also for aLRP Regression error
            rank[ii] = rank_pos + FP_num

            # Compute precision for this example to compute classification loss
            prec[ii] = rank_pos / rank[ii]
            # For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:
                fg_grad[ii] = -(torch.sum(fg_relations * regression_losses) + FP_num) / rank[ii]
                relevant_bg_grad += (bg_relations * (-fg_grad[ii] / FP_num))

                # aLRP with grad formulation fg gradient
        classification_grads[fg_labels] = fg_grad
        # aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels] = relevant_bg_grad

        classification_grads /= (fg_num)

        cls_loss = 1 - prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None, None, None


class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.):
        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta

        # Get valid bg logits
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))
        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        rank = torch.zeros(fg_num).cuda()
        prec = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        max_prec = 0
        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # x_ij s as score differences with fgs
            fg_relations = fg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with fgs
            fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)
            # Discard i=j in the summation in rank_pos
            fg_relations[ii] = 0

            # x_ij s as score differences with bgs
            bg_relations = relevant_bg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with bgs
            bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)

            # Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos = 1 + torch.sum(fg_relations)
            FP_num = torch.sum(bg_relations)
            # Store the total since it is normalizer also for aLRP Regression error
            rank[ii] = rank_pos + FP_num

            # Compute precision for this example
            current_prec = rank_pos / rank[ii]

            # Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec <= current_prec):
                max_prec = current_prec
                relevant_bg_grad += (bg_relations / rank[ii])
            else:
                relevant_bg_grad += (bg_relations / rank[ii]) * (((1 - max_prec) / (1 - current_prec)))

            # Store fg gradients
            fg_grad[ii] = -(1 - max_prec)
            prec[ii] = max_prec

            # aLRP with grad formulation fg gradient
        classification_grads[fg_labels] = fg_grad
        # aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels] = relevant_bg_grad

        classification_grads /= fg_num

        cls_loss = 1 - prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None