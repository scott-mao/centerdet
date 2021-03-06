from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, re-sampling from it until
    a specified number of iterations have been sampled
    """
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        # super(IterationBasedBatchSampler, self).__init__()
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler.data_source.pipeline.gridmask, 'set_prob'):
                self.batch_sampler.sampler.data_source.pipeline.gridmask.set_prob(iteration, self.num_iterations)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations