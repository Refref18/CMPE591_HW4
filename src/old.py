import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import environment
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create test_results folder if it does not exist
if not os.path.exists("test_results"):
    os.makedirs("test_results")
class CNP(torch.nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve

def make_batch_with_indices(traj, max_context, max_target):
    L = traj.shape[0]
    n_c = random.randint(1, min(max_context, L))
    n_t = 1
    idx_perm = np.random.permutation(L)
    idx_c = np.sort(idx_perm[:n_c])
    idx_t = np.sort(idx_perm[n_c:n_c + n_t])

    obs_np   = traj[idx_c]        # (n_c, 6)
    targ_np  = traj[idx_t][:, [0, 5]]
    truth_np = traj[idx_t, 1:5]   # (n_t, 4)

    obs = torch.from_numpy(obs_np).float().unsqueeze(0)
    targ = torch.from_numpy(targ_np).float().unsqueeze(0)
    truth = torch.from_numpy(truth_np).float().unsqueeze(0)
    return obs, targ, truth, idx_c, idx_t

def make_batch(traj, max_context, max_target ):
    """
    traj: np.array (L,6): [t, ey, ez, oy, oz, h]
    """
    L = len(traj)
    if L < 2:
        raise ValueError("traj too short")

    # 1) pick a RANDOM number of context pts between 1 and min(max_context, L-1)
    n_c = max_context
    # 2) pick a RANDOM number of target pts between 1 and min(max_target, L-n_c)
    n_t = 1

    # 3) shuffle and split
    perm = np.random.permutation(L)
    idx_c = perm[:n_c]                    # take first n_c shuffled indices
    idx_t = perm[n_c:n_c + n_t]           # then next n_t for targets
    # Slice numpy arrays
    obs_np   = traj[idx_c]        # (n_c, 6)
    targ_np  = traj[idx_t][:, [0, 5]]
    truth_np = traj[idx_t, 1:5]   # (n_t, 4)

    # Convert to torch tensors and add batch dimension
    obs   = torch.from_numpy(obs_np)  .float().unsqueeze(0)
    targ  = torch.from_numpy(targ_np) .float().unsqueeze(0)
    truth = torch.from_numpy(truth_np).float().unsqueeze(0)
    return obs, targ, truth

if __name__ == "__main__":
    env = Hw5Env(render_mode="offscreen")
    #    d_x = 2 (t and h), d_y = 4 (ey, ez, oy, oz)
    d_x, d_y = 2, 4

    hidden_size       = 128
    num_hidden_layers = 3
    lr                = 5e-5
    
    n_context = random.randint(1, 10)
    n_target  = 1

    model = CNP(
        in_shape=(d_x, d_y),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        min_std=0.1
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    states_arr = []
    num_trajectories = 100

    # Collect data
    for i in range(num_trajectories):
        
        env.reset()
        p1 = np.array([0.5,  0.3, 1.04])
        p2 = np.array([0.5,  0.15, np.random.uniform(1.04, 1.4)])
        p3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p1, p2, p3, p4], axis=0)
        curve  = bezier(points)    # shape (L,3)

        env._set_ee_in_cartesian(curve[0],
                                rotation=[-90,0,180],
                                n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90,0,180], max_iters=10)
            hl = env.high_level_state()   # → np.array([ey, ez, oy, oz, h])
            states.append(hl)

        states = np.stack(states)  
        states_arr.append(states)

        print(f"Collected {i+1}/{num_trajectories} trajectories.", end="\r")
    # 1) Convert each (L×5) states array into an (L×6) trajectory:
    trajectories = []
    for states in states_arr:
        L = states.shape[0]
        # a) normalized time t from 0→1
        t = np.linspace(0, 1, L).reshape(-1, 1)   # shape (L,1)

        # b) the height h (constant per trajectory)
        h = states[0, 4]                          # the last column of states
        h_col = np.ones((L, 1)) * h               # shape (L,1)

        # c) stack into [t, ey, ez, oy, oz, h]
        traj = np.concatenate([t, states[:, 0:4], h_col], axis=1)
        #print(traj[0])
        trajectories.append(traj)                 # now traj.shape == (L,6)
    
    print("Built", len(trajectories), "trajectories with time+height.")
    obs, targ, truth = make_batch(trajectories[0], n_context, n_target)
    print(obs.shape, targ.shape, truth.shape)
    # --- TRAINING LOOP ---
    n_epochs = 100000

    batch_size = 32

    for epoch in range(1, n_epochs+1):
        model.train()

        # 1) sample batch_size trajectories
        batch_trajs = random.sample(trajectories, batch_size)

        obs_list, targ_list, truth_list = [], [], []

        # 2) build each context/target
        for traj in batch_trajs:
            obs, targ, truth = make_batch(traj, n_context, n_target)
            obs_list.append(obs)       # each is shape (1, n_c, d_x+d_y)
            targ_list.append(targ)     # each is shape (1, n_t, d_x)
            truth_list.append(truth)   # each is shape (1, n_t, d_y)

        # 3) stack into batch tensors
        # result shapes: (32, n_c, d_x+d_y), (32, n_t, d_x), (32, n_t, d_y)
        obs_batch   = torch.cat(obs_list,   dim=0)
        targ_batch  = torch.cat(targ_list,  dim=0)
        truth_batch = torch.cat(truth_list, dim=0)

        # 4) single forward/backward on the whole batch
        loss = model.nll_loss(obs_batch, targ_batch, truth_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} — batch NLL: {loss.item():.4f}")
    model.eval()
    max_context = 10
    mse_e, mse_o = [], []

    with torch.no_grad():
        for _ in range(100):
            traj = random.choice(trajectories)
            
            # her seferinde 1 ile 200 arasında rastgele context
            n_c_rand = random.randint(1, max_context)
            # target sayısı hâlâ istediğin sabit (örneğin 1) kalabilir
            n_t_rand = n_target   

            obs, targ, truth = make_batch(traj, n_c_rand, n_t_rand)

            mean, _ = model(obs, targ)
            pred = mean.squeeze(0).cpu().numpy()
            gt   = truth.squeeze(0).cpu().numpy()

            mse_e.append(np.mean((pred[:, :2] - gt[:, :2])**2))
            mse_o.append(np.mean((pred[:, 2:] - gt[:, 2:])**2))

    # ortalama ve sapma
    mean_e, std_e = np.mean(mse_e), np.std(mse_e)
    mean_o, std_o = np.mean(mse_o), np.std(mse_o)
    
    print(f"End‑Effector MSE: {mean_e:.5f} ± {std_e:.5f}")
    print(f"Object        MSE: {mean_o:.5f} ± {std_o:.5f}")

    # 4) Bar plot
    plt.figure()
    plt.bar(
        ["End‑Effector","Object"],
        [mean_e, mean_o],
        yerr=[std_e, std_o],
        capsize=5
    )
    plt.ylabel("MSE")
    plt.title("100‑Test Mean ± Std MSE")
    plt.show()

    # Plot 100 tests
    for test_i in range(100):
        traj = random.choice(trajectories)
        obs, targ, truth, idx_c, idx_t = make_batch_with_indices(traj, max_context, n_target)

        model.eval()
        with torch.no_grad():
            mean, _ = model(obs, targ)
        pred = mean.squeeze(0).cpu().numpy()      # (n_t, 4)
        gt = truth.squeeze(0).cpu().numpy()       # (n_t, 4)
        print(model(obs, targ),pred,gt)

        # For plotting, take only the first target (since n_t=1)
        tc = idx_c  # context indices
        tt = idx_t[0]  # target index
        pred_e_y, pred_e_z, pred_o_y, pred_o_z = pred[0]
        gt_e_y,   gt_e_z,   gt_o_y,   gt_o_z = gt[0]

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # 1) End‑effector: show full trajectory as gray points
        axes[0].scatter(
            traj[:,1], traj[:,2],
            c='lightgray', alpha=0.3, s=10, label='Full Trajectory'
        )
        # then overplot the line if you like
        axes[0].plot(traj[:,1], traj[:,2], 'k-', alpha=0.2)
        # context
        axes[0].scatter(
            traj[tc,1], traj[tc,2],
            c='blue', label='Context'
        )
        # true
        axes[0].scatter(
            traj[tt,1], traj[tt,2],
            c='green', label='True Target'
        )
        # predicted
        axes[0].scatter(
            pred_e_y, pred_e_z,
            c='red', marker='x', s=100, label='Predicted'
        )
        axes[0].set_title(f"Test {test_i+1} — End‑Effector")
        axes[0].set_xlabel("e_y")
        axes[0].set_ylabel("e_z")
        axes[0].legend()

        # 2) Object: same pattern
        axes[1].scatter(
            traj[:,3], traj[:,4],
            c='lightgray', alpha=0.3, s=10, label='Full Trajectory'
        )
        axes[1].plot(traj[:,3], traj[:,4], 'k-', alpha=0.2)
        axes[1].scatter(
            traj[tc,3], traj[tc,4],
            c='blue', label='Context'
        )
        axes[1].scatter(
            traj[tt,3], traj[tt,4],
            c='green', label='True Target'
        )
        axes[1].scatter(
            pred_o_y, pred_o_z,
            c='red', marker='x', s=100, label='Predicted'
        )
        axes[1].set_title(f"Test {test_i+1} — Object")
        axes[1].set_xlabel("o_y")
        axes[1].set_ylabel("o_z")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"test_results/test_{test_i+1}.png")

