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
        self.d_x = in_shape[0]  # query input (t)
        self.d_y = in_shape[1]  # target output (e_y, e_z, o_y, o_z)

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y + 1, hidden_size))  # +1 for height (h)
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x + 1, hidden_size))  # +1 for height (h)
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))  # mean + std
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def forward(self, observation, target, condition, observation_mask=None):
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target, condition)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def nll_loss(self, observation, target, target_truth, condition, observation_mask=None, target_mask=None):
        mean, std = self.forward(observation, target, condition, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def encode(self, observation):
        return self.encoder(observation)

    def decode(self, h):
        return self.query(h)

    def aggregate(self, h, observation_mask=None):
        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)
            return h / normalizer
        else:
            return h.mean(dim=1)

    def concatenate(self, r, target, condition):
        B, N, _ = target.shape
        condition = condition.unsqueeze(1).repeat(1, N, 1)  # [B, N, 1]
        r = r.unsqueeze(1).repeat(1, N, 1)  # [B, N, H]
        return torch.cat([r, target, condition], dim=-1)




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

def prepare_batch(states_arr, n_batch=16, t_scale=100.0):
    """
    Prepare a random training batch for CNMP.

    Returns:
        obs:         [n_batch, n_context, d_x + d_y]
        target:      [n_batch, n_target, d_x]
        target_y:    [n_batch, n_target, d_y]
        condition_h: [n_batch, 1]  # same h for each sample in batch
    """
    obs_list, target_list, target_y_list, h_list = [], [], [], []

    for _ in range(n_batch):
        traj = states_arr[np.random.randint(len(states_arr))]  # [T, 5]
        T = traj.shape[0]
        t_vals = np.linspace(0, 1, T).reshape(-1, 1)  # Normalize time t between 0 and 1
        full_traj = np.concatenate([t_vals, traj], axis=1)  # [T, 6]: [t, e_y, e_z, o_y, o_z, h]

        # Random number of context & target points
        idxs = np.random.permutation(T)
        n_context = np.random.randint(3, T // 2)
        n_target = np.random.randint(3, T - n_context)

        obs_idx = idxs[:n_context]
        target_idx = idxs[n_context:n_context + n_target]

        obs = full_traj[obs_idx, :]  # Include all 6 features: [t, e_y, e_z, o_y, o_z, h]
        target = full_traj[target_idx, [0]].reshape(-1, 1)  # Add an extra dimension for d_x
        target_y = full_traj[target_idx, 1:5]  # [e_y, e_z, o_y, o_z] — ground truth

        h_val = full_traj[0, 5]  # same height for all

        obs_list.append(torch.tensor(obs, dtype=torch.float32))
        target_list.append(torch.tensor(target, dtype=torch.float32))
        target_y_list.append(torch.tensor(target_y, dtype=torch.float32))
        h_list.append(torch.tensor([h_val], dtype=torch.float32))

    # Pad and stack if needed
    return (
        torch.nn.utils.rnn.pad_sequence(obs_list, batch_first=True),       # [B, N_ctx, dx+dy]
        torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True),    # [B, N_tgt, dx]
        torch.nn.utils.rnn.pad_sequence(target_y_list, batch_first=True),  # [B, N_tgt, dy]
        torch.stack(h_list)                                                # [B, 1]
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
    #plt.show()
    # 🧠 Define model and optimizer here before using them!
    # Model parametreleri
    input_dim = 1     # sadece t
    output_dim = 4    # ey, ez, oy, oz
    hidden_size = 512
    num_layers = 3

    # Model ve optimizer
    model = CNP(in_shape=(input_dim, output_dim), hidden_size=hidden_size, num_hidden_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Eğitim
    model.train()
    for epoch in range(1000):  # çoklu epoch istersen
        obs, target_x, target_y, condition = prepare_batch(states_arr)
        optimizer.zero_grad()
        loss = model.nll_loss(obs, target_x, target_y, condition)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:02d} - Loss: {loss.item():.4f}")

    print(condition)
    model.eval()
    with torch.no_grad():
        obs, target_x, target_y, condition = prepare_batch(states_arr, n_batch=1)
        mean, std = model(obs, target_x, condition)

    mean_pred = mean[0].numpy()
    ground_truth = target_y[0].numpy()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], color='green', label='True End-Effector')
    plt.scatter(mean_pred[:, 0], mean_pred[:, 1], color='blue', label='Predicted End-Effector')
    plt.xlabel("e_y")
    plt.ylabel("e_z")
    plt.legend()
    plt.title("End-Effector Prediction")

    plt.subplot(1, 2, 2)
    plt.scatter(ground_truth[:, 2], ground_truth[:, 3], color='red', label='True Object')
    plt.scatter(mean_pred[:, 2], mean_pred[:, 3], color='orange', label='Predicted Object')
    plt.xlabel("o_y")
    plt.ylabel("o_z")
    plt.legend()
    plt.title("Object Prediction")

    plt.tight_layout()
    plt.show()
