import torch

class Metric:
    """
        Metric class
    """
    @staticmethod
    def r_q(tt, pred_time, epsilon):
        """
            returns R(q) test loss defined in the QPP paper
        """
        rq_vec, _ = torch.max(
            torch.cat([((tt+epsilon)/(pred_time+epsilon)).unsqueeze(0),
                       ((pred_time+epsilon)/(tt+epsilon)).unsqueeze(0)], axis=0),
            axis=0)
        # print(rq_vec.shape)
        curr_rq = torch.mean(rq_vec).item()
        return curr_rq

    @staticmethod
    def pred_err(tt, pred_time, epsilon):
        """
            returns a vector of pred_err for each sample in the input
        """
        curr_pred_err = (torch.abs(tt - pred_time) + epsilon) / (tt + epsilon)
        return curr_pred_err

    @staticmethod
    def accumulate_err(tt, pred_time, epsilon):
        """
            returns the pred_err for the sum of predictions
        """
        tt_sum = torch.sum(tt)
        pred_time_sum = torch.sum(pred_time)
        return torch.abs(pred_time_sum - tt_sum + epsilon) / (tt_sum + epsilon)

    @staticmethod
    def mean_mae(tt, pred_time, epsilon):
        """
            returns the absolute error for the mean of predictions
        """
        tt_mean = torch.mean(tt)
        pred_time_mean = torch.mean(pred_time)
        return torch.abs(pred_time_mean - tt_mean)
