import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import environment
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

def prepare_batch(states_arr, n_batch=1):
    obs_list, target_x_list, target_y_list, h_list = [], [], [], []

    for _ in range(n_batch):
        traj = states_arr[np.random.randint(len(states_arr))]  # [T, 5]
        T = traj.shape[0]
        t_vals = np.linspace(0, 1, T).reshape(-1, 1)  # [T, 1]

        full_traj = np.concatenate([t_vals, traj], axis=1)  # [T, 6] → (t, ey, ez, oy, oz, h)

        idxs = np.random.permutation(T)
        n_context = np.random.randint(3, T // 2)
        n_target = np.random.randint(3, T - n_context)

        obs_idx = idxs[:n_context]
        target_idx = idxs[n_context:n_context + n_target]

        obs = full_traj[obs_idx, :]        # Include all 6 features (t, ey, ez, oy, oz, h)
        target_x = full_traj[target_idx][:, [0, 5]]  # (t, h)
        target_y = full_traj[target_idx][:, 1:5]     # (ey, ez, oy, oz)

        h_val = full_traj[0, 5]  # Aynı h tüm noktalar için

        obs_list.append(torch.tensor(obs, dtype=torch.float32))
        target_x_list.append(torch.tensor(target_x, dtype=torch.float32))
        target_y_list.append(torch.tensor(target_y, dtype=torch.float32))
        h_list.append(torch.tensor([h_val], dtype=torch.float32))

    return (
        torch.nn.utils.rnn.pad_sequence(obs_list, batch_first=True),     # [B, N_ctx, 6]
        torch.nn.utils.rnn.pad_sequence(target_x_list, batch_first=True),# [B, N_tgt, 2]
        torch.nn.utils.rnn.pad_sequence(target_y_list, batch_first=True),# [B, N_tgt, 4]
        torch.stack(h_list)                                              # [B, 1]
    )


if __name__ == "__main__":
    env = Hw5Env(render_mode="offscreen")
    states_arr = []
    for i in range(100):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        states = np.stack(states)
        states_arr.append(states)
        print(f"Collected {i+1} trajectories.", end="\r")

    fig, ax = plt.subplots(1, 2)
    for states in states_arr:
        ax[0].plot(states[:, 0], states[:, 1], alpha=0.2, color="b")
        ax[0].set_xlabel("e_y")
        ax[0].set_ylabel("e_z")
        ax[1].plot(states[:, 2], states[:, 3], alpha=0.2, color="r")
        ax[1].set_xlabel("o_y")
        ax[1].set_ylabel("o_z")
    plt.show()
    """# Her trajectory'den 10 örnek göster
    for traj_id, states in enumerate(states_arr):
        T = states.shape[0]
        t_vals = np.linspace(0, 1, T).reshape(-1, 1)  # t: 0 → 1 arası
        full_traj = np.concatenate([t_vals, states], axis=1)  # [t, e_y, e_z, o_y, o_z, h]

        print(f"\nTrajectory {traj_id + 1} samples:")
        sample_idxs = np.linspace(0, T - 1, 10, dtype=int)  # 10 eşit aralıklı nokta al
        for idx in sample_idxs:
            t, ey, ez, oy, oz, h = full_traj[idx]
            print(f"  t={t:.2f}, ey={ey:.3f}, ez={ez:.3f}, oy={oy:.3f}, oz={oz:.3f}, h={h:.3f}")"""
        
    # ------------------------------------------------------------
    # 1) Modeli tanımla
    #    d_x = 2  -> (t, h)           d_y = 4  -> (ey, ez, oy, oz)
    # ------------------------------------------------------------
    cnp = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=3)
    optimizer = torch.optim.Adam(cnp.parameters(), lr=1e-3)

    # ------------------------------------------------------------
    # 2) Eğitim döngüsü (yalnızca toplanan trajelerle)
    # ------------------------------------------------------------
    n_epochs = 500
    batch_size = 4

    for epoch in range(1, n_epochs + 1):
        # Tek batch hazırlayıp hemen geri yayılım
        obs, tgt_x, tgt_y, _ = prepare_batch(states_arr, n_batch=batch_size)

        loss = cnp.nll_loss(obs, tgt_x, tgt_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"[{epoch}/{n_epochs}]  NLL loss: {loss.item():.4f}")

    # ------------------------------------------------------------
    # 3) Test – toplam 5 nokta seçip tahmin et
    # ------------------------------------------------------------
    test_traj = states_arr[np.random.randint(len(states_arr))]
    T = test_traj.shape[0]
    t_vals = np.linspace(0, 1, T).reshape(-1, 1)
    full_traj = np.concatenate([t_vals, test_traj], axis=1)   # [t, ey, ez, oy, oz, h]

    idxs = np.random.permutation(T)
    ctx_idx, tgt_idx = idxs[5:], idxs[:5]                     # 5 hedef, geri kalanı context

    obs      = torch.tensor(full_traj[ctx_idx], dtype=torch.float32).unsqueeze(0)      # [1, N_ctx, 6]
    target_x = torch.tensor(full_traj[tgt_idx][:, [0, 5]], dtype=torch.float32).unsqueeze(0)  # [1, 5, 2]
    target_y = torch.tensor(full_traj[tgt_idx][:, 1:5],  dtype=torch.float32).unsqueeze(0)    # [1, 5, 4]

    with torch.no_grad():
        pred_mean, pred_std = cnp(obs, target_x)

    pred_mean = pred_mean.squeeze(0).numpy()   # [5, 4]
    target_y  = target_y.squeeze(0).numpy()     # [5, 4]
    tgt_t     = target_x.squeeze(0)[:, 0].numpy()

    # ------------------------------------------------------------
    # 4) Görselleştir – her boyutu ayrı subplot’ta
    # ------------------------------------------------------------
    labels = ["e_y", "e_z", "o_y", "o_z"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.errorbar(tgt_t, pred_mean[:, i], yerr=pred_std.squeeze(0)[:, i],
                    fmt="o", label="Prediction ± σ")
        ax.scatter(tgt_t, target_y[:, i], color="red", label="Ground truth")
        ax.set_xlabel("t")
        ax.set_ylabel(labels[i])
        ax.set_title(f"Predicted vs True – {labels[i]}")
        ax.grid(True)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


    # ------------------------------------------------------------
    # 5) Run 100 random test‐batches and compute MSE
    # ------------------------------------------------------------
    n_tests = 100
    mse_end_list = []
    mse_obj_list = []

    for _ in range(n_tests):
        # --- 5.1 Generate one fresh trajectory ---
        env.reset()
        # pick 4 control points (varying the middle two randomly)
        p1 = np.array([0.5,  0.3,  1.04])
        p2 = np.array([0.5,  0.15, np.random.uniform(1.04, 1.4)])
        p3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p4 = np.array([0.5, -0.3,  1.04])
        curve = bezier(np.stack([p1, p2, p3, p4], axis=0))
        # record high_level_state at each step
        states = []
        env._set_ee_in_cartesian(curve[0], rotation=[-90,0,180], n_splits=100, max_iters=100, threshold=0.05)
        for pt in curve:
            env._set_ee_pose(pt, rotation=[-90,0,180], max_iters=10)
            states.append(env.high_level_state())
        states = np.stack(states)             # shape [T, 5]

        # --- 5.2 Build full_traj with time stamps ---
        T = states.shape[0]
        t_vals = np.linspace(0, 1, T).reshape(-1, 1)
        full_traj = np.concatenate([t_vals, states], axis=1)  # [T, 6]

        # --- 5.3 Random context/target split ---
        idxs      = np.random.permutation(T)
        n_context = np.random.randint(3, T//2)
        n_target  = np.random.randint(3, T - n_context)
        obs_idx   = idxs[:n_context]
        tgt_idx   = idxs[n_context:n_context + n_target]

        # --- 5.4 Extract arrays and convert to batch‐shape tensors ---
        obs_np    = full_traj[obs_idx]                  # [N_ctx, 6]
        tx_np     = full_traj[tgt_idx][:, [0,5]]        # [N_tgt, 2]
        ty_np     = full_traj[tgt_idx][:, 1:5]          # [N_tgt, 4]

        obs_tensor    = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)   # [1, N_ctx, 6]
        target_x_tensor = torch.tensor(tx_np, dtype=torch.float32).unsqueeze(0)  # [1, N_tgt, 2]
        target_y_tensor = torch.tensor(ty_np, dtype=torch.float32).unsqueeze(0)  # [1, N_tgt, 4]

        # --- 5.5 Predict on this fresh batch ---
        with torch.no_grad():
            pred_mean, _ = cnp(obs_tensor, target_x_tensor)

        pred = pred_mean.squeeze(0).numpy()    # [N_tgt, 4]
        true = ty_np                           # [N_tgt, 4]

        # --- 5.6 Compute MSE separately ---
        se = (pred - true)**2
        mse_end_list.append(se[:, :2].mean())  # dims 0–1 = end‑effector
        mse_obj_list.append(se[:, 2:].mean())  # dims 2–3 = object

        # --- 5.7 Visualize this test’s predictions vs ground truth ---
        labels = ["e_y", "e_z", "o_y", "o_z"]
        tgt_t = tx_np[:, 0]  # the time values for each target point

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            # plot prediction ± σ if you have pred_std available:
            # ax.errorbar(tgt_t, pred[:, i], yerr=pred_std_np[:, i], fmt="o", label="Pred ± σ")
            ax.plot(tgt_t, pred[:, i], 'o', label="Prediction")
            ax.scatter(tgt_t, true[:, i], color="red", label="Ground truth")
            ax.set_xlabel("t")
            ax.set_ylabel(labels[i])
            ax.set_title(f"Pred vs True – {labels[i]}")
            ax.grid(True)
            ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 6) Bar plot of mean MSE ± std over these 100 fresh tests
    # ------------------------------------------------------------
    mse_end = np.array(mse_end_list)
    mse_obj = np.array(mse_obj_list)
    means = [mse_end.mean(), mse_obj.mean()]
    stds  = [mse_end.std(),  mse_obj.std()]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(
        ["End‑effector", "Object"],
        means,
        yerr=stds,
        capsize=5
    )
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("CNP MSE on 100 Randomly Generated Trajectories")
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.show()