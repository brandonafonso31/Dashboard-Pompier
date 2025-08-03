import torch
from agent import POMO_Agent
import torch.optim as optim

def generate_instances(batch_size, n_nodes):
    return torch.rand(batch_size, n_nodes, 2)

def compute_tour_length(coords, tours):
    # coords: [B, N, 2], tours: [B, pomo, N]
    B, P, N = tours.size()
    tour_coords = torch.gather(coords.unsqueeze(1).expand(-1, P, -1, -1), 2, tours.unsqueeze(-1).expand(-1, -1, -1, 2))
    rolled = torch.roll(tour_coords, shifts=-1, dims=2)
    lengths = ((tour_coords - rolled) ** 2).sum(dim=-1).sqrt().sum(dim=-1)  # [B, P]
    return lengths

N_EPOCHS = 1000
BATCH_SIZE = 64
N_NODES = 20
POMO_SIZE = 2  # Nombre de tours POMO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = POMO_Agent().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(N_EPOCHS):
    coords = generate_instances(BATCH_SIZE, N_NODES).to(device)
    B = coords.size(0)

    node_embeds = model.encoder(coords)  # [B, N, H]
    rewards = []
    log_probs = []

    for _ in range(POMO_SIZE):
        mask = torch.zeros(B, N_NODES, device=device)
        tour = []
        log_p = []
        for _ in range(N_NODES):
            probs = model(node_embeds, mask)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            mask[torch.arange(B), action] = 1
            tour.append(action)
            log_p.append(m.log_prob(action))
        tour = torch.stack(tour, dim=1)  # [B, N]
        log_p = torch.stack(log_p, dim=1)  # [B, N]
        log_probs.append(log_p.sum(dim=1))  # [B]
        rewards.append(compute_tour_length(coords, tour.unsqueeze(1)).squeeze(1))  # [B]

    log_probs = torch.stack(log_probs, dim=1)  # [B, P]
    rewards = torch.stack(rewards, dim=1)  # [B, P]

    # baseline POMO: moyenne sur chaque instance
    baseline = rewards.mean(dim=1, keepdim=True)
    advantage = baseline - rewards  # minimiser la distance => max(-length)
    loss = (advantage.detach() * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Avg reward: {rewards.min(dim=1)[0].mean():.4f}")