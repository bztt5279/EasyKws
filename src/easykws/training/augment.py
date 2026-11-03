"""Feature-domain masks on a clone, with restorable Python random state."""
import random

class SpecAugment:
    def __init__(self, num_t_mask=1, num_f_mask=1, max_t=20, max_f=10):
        for key, value in locals().copy().items():
            if key != 'self' and (type(value) is not int or value < 0):
                raise ValueError('Mask counts and bounds must be nonnegative integers')
        if num_t_mask and not max_t or num_f_mask and not max_f:
            raise ValueError('Enabled mask bounds must be positive')
        self.num_t_mask, self.num_f_mask, self.max_t, self.max_f = num_t_mask, num_f_mask, max_t, max_f

    def __call__(self, features):
        if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
            raise ValueError('Features must be nonempty [frames, bins]')
        result = features.clone()
        for axis, count, maximum in ((0, self.num_t_mask, self.max_t), (1, self.num_f_mask, self.max_f)):
            for _ in range(count):
                start = random.randint(0, result.shape[axis] - 1)
                end = min(result.shape[axis], start + random.randint(1, maximum))
                if axis == 0:
                    result[start:end, :] = 0
                else:
                    result[:, start:end] = 0
        return result
