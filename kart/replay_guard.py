"""Real recording rehearsal and a separate, immutable policy regression gate."""
import numpy as np
import torch
import torch.nn.functional as F


class ReplayGuard:
    def __init__(self, path, device):
        with np.load(path, allow_pickle=False) as src:
            self.data = {}
            for group in ['train', 'validation', 'critical']:
                grid = torch.as_tensor(src[group+'_grid'], device=device, dtype=torch.float32)[:, None]/2
                state = torch.as_tensor(src[group+'_state'], device=device, dtype=torch.float32)
                target = torch.as_tensor(src[group+'_target'], device=device, dtype=torch.long)
                if (not len(target) or grid.shape != (len(target), 1, 48, 48)
                        or state.shape != (len(target), 8) or not torch.isfinite(grid).all()
                        or not torch.isfinite(state).all() or not ((target == 0) | (target == 1)).all()):
                    raise ValueError('Invalid recording regression bank: '+group)
                self.data[group] = (grid, state, target)
            self.minimum_accuracy = float(src['minimum_accuracy'])
            if not .9 <= self.minimum_accuracy <= 1.:
                raise ValueError('Invalid recording gate accuracy threshold')

    def loss(self, net, batch):
        # Only the original training split may affect gradients. Validation and
        # critical windows are never rehearsal targets.
        grid, state, target = self.data['train']
        ids = torch.randint(len(target), (batch,), device=grid.device)
        logits, _ = net(grid[ids], state[ids])
        return F.cross_entropy(logits, target[ids])

    @torch.inference_mode()
    def evaluate(self, net):
        result = {}
        for group in ['validation', 'critical']:
            grid, state, target = self.data[group]
            logits = torch.cat([net(grid[i:i+256], state[i:i+256])[0] for i in range(0, len(target), 256)])
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite recording evaluation logits')
            wrong = int((logits.argmax(1) != target).sum().item())
            result[group] = dict(count=len(target), errors=wrong, accuracy=1-wrong/len(target))
        result['minimum_accuracy'] = self.minimum_accuracy
        result['passed'] = (result['validation']['accuracy'] >= self.minimum_accuracy
                            and result['critical']['errors'] == 0)
        return result
