import networkx as nx
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from torch.distributions import Categorical
from tqdm import tqdm
import community.community_louvain as community
import warnings
from collections import deque
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(filename='training.log', level=logging.INFO)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Determine the device to use (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Graph Initialization
# -------------------------
def initialize_graph(num_nodes=10000, m=5):
    graph = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=42)
    return graph

# -------------------------
# Feature Extraction
# -------------------------
def extract_features(subgraph, controlled_nodes, adversarial_nodes, node_resource_costs, global_state, stability_history, clustering_coeff):
    feature_matrix = []
    node_list = list(subgraph.nodes())

    for node in node_list:
        is_controlled = 1 if node in controlled_nodes else 0
        is_adversarial = 1 if node in adversarial_nodes else 0
        resource_cost = node_resource_costs.get(node, 1.0)
        degree_centrality = subgraph.degree(node)
        clustering = clustering_coeff.get(node, 0)
        stability = stability_history.get(node, 0)
        global_controlled = global_state.get('total_controlled', 0)

        feature_vector = [
            resource_cost,
            is_controlled,
            is_adversarial,
            degree_centrality,
            clustering,
            stability,
            global_controlled,
        ]

        feature_matrix.append(feature_vector)

    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float).to(device)
    return feature_matrix, node_list

# -------------------------
# Centrality Measures
# -------------------------
def compute_centrality_measures(graph):
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = degree_centrality
    closeness_centrality = nx.closeness_centrality(graph)
    try:
        katz_centrality = nx.katz_centrality(graph, alpha=0.005, beta=1.0)
    except nx.PowerIterationFailedConvergence:
        katz_centrality = degree_centrality
    return degree_centrality, betweenness_centrality, eigenvector_centrality, closeness_centrality, katz_centrality

# -------------------------
# Resource Cost Initialization
# -------------------------
def initialize_resource_costs(graph, degree_centrality, betweenness_centrality,
                              eigenvector_centrality, closeness_centrality, katz_centrality):
    resource_costs = {}
    for node in graph.nodes:
        cost = (degree_centrality[node] * 1.0 +
                betweenness_centrality[node] * 2.0 +
                eigenvector_centrality[node] * 1.5 +
                closeness_centrality[node] * 1.0 +
                katz_centrality[node] * 1.2)
        normalized_cost = (cost / (degree_centrality[node] + betweenness_centrality[node] +
                                    eigenvector_centrality[node] + closeness_centrality[node] +
                                    katz_centrality[node])) + random.uniform(-0.1, 0.1)
        normalized_cost = max(0.1, normalized_cost)
        resource_costs[node] = normalized_cost
    return resource_costs

# -------------------------
# Graph Evolution
# -------------------------
def evolve_graph(graph, degree_centrality, agent_success, adversary_success, p_add_base=0.001, p_remove_base=0.001, m=5):
    """
    Adaptive graph evolution based on agent and adversary success metrics.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if agent_success > adversary_success:
        p_add = p_add_base * 1.5
        p_remove = p_remove_base * 0.5
    elif adversary_success > agent_success:
        p_add = p_add_base * 0.5
        p_remove = p_remove_base * 1.5
    else:
        p_add = p_add_base
        p_remove = p_remove_base

    # Add edges
    for _ in range(int(num_edges * p_add)):
        u, v = random.choices(list(graph.nodes()), weights=[degree_centrality.get(node, 1) for node in graph.nodes()], k=2)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

    # Remove edges
    edges = list(graph.edges())
    for edge in edges:
        if random.random() < p_remove:
            graph.remove_edge(*edge)

    # Add nodes
    if random.random() < p_add:
        new_node = num_nodes
        graph.add_node(new_node)
        targets = random.choices(list(graph.nodes()), weights=[degree_centrality.get(node, 1) for node in graph.nodes()], k=m)
        for target in targets:
            graph.add_edge(new_node, target)

    # Remove nodes
    if random.random() < p_remove and num_nodes > 1:
        remove_node = random.choice(list(graph.nodes()))
        graph.remove_node(remove_node)

    return graph

# -------------------------
# Neural Network Models
# -------------------------
class DetectionProbabilityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(DetectionProbabilityPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(x), dim=1)
        return prob

class GNNPolicyNetworkWithMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_hidden_dim, num_heads=4):
        super(GNNPolicyNetworkWithMemory, self).__init__()

        self.gnn = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.gnn2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, data, memory):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gnn(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gnn2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add time dimension for LSTM
        lstm_out, memory = self.lstm(x, memory)
        lstm_out = lstm_out.squeeze(1)  # Remove time dimension
        x = self.fc(lstm_out)
        action_probs = F.softmax(x, dim=1)
        return action_probs, memory

class AdversaryActionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3):
        super(AdversaryActionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_pred = F.softmax(self.fc2(x), dim=1)
        return action_pred

# -------------------------
# Predictor Initialization
# -------------------------
def initialize_adversary_predictor(input_dim, hidden_dim, num_classes=3, lr=0.001):
    predictor = AdversaryActionPredictor(input_dim, hidden_dim, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(predictor.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    return predictor, optimizer, scheduler

def initialize_detection_predictor(input_dim, hidden_dim, num_classes=2, lr=0.001):
    detector = DetectionProbabilityPredictor(input_dim, hidden_dim, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(detector.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    return detector, optimizer, scheduler

# -------------------------
# Training Functions
# -------------------------
def train_adversary_predictor(predictor, optimizer, scheduler, features, adversary_actions):
    predictor.train()
    optimizer.zero_grad()
    action_preds = predictor(features)
    loss = F.cross_entropy(action_preds, adversary_actions)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def train_detection_predictor(detector, optimizer, scheduler, features, labels):
    detector.train()
    optimizer.zero_grad()
    predictions = detector(features)
    loss = F.cross_entropy(predictions, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

# -------------------------
# Agent Class
# -------------------------
class AgentWithMemory:
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_hidden_dim, lr=0.001, entropy_coeff=0.01):
        self.policy_net = GNNPolicyNetworkWithMemory(input_dim, hidden_dim, output_dim, lstm_hidden_dim).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.memory = None
        self.entropy_coeff = entropy_coeff

    def reset_memory(self):
        self.memory = (torch.zeros(1, 1, self.policy_net.lstm.hidden_size).to(device),
                       torch.zeros(1, 1, self.policy_net.lstm.hidden_size).to(device))

    def select_actions(self, data, available_actions, num_actions, epsilon=0.1):
        self.policy_net.eval()
        with torch.no_grad():
            action_probs, self.memory = self.policy_net(data, self.memory)
            mask = torch.zeros_like(action_probs).to(device)
            mask[0, available_actions] = 1
            masked_probs = action_probs * mask

            if masked_probs.sum() == 0:
                # Fallback to heuristic: select the node with the highest degree
                node_indices = available_actions
                degrees = [graph.degree(node_list[i]) for i in node_indices]
                max_degree_idx = node_indices[np.argmax(degrees)]
                return [max_degree_idx], torch.log(torch.tensor([1.0])).to(device)

            if random.random() < epsilon:
                # Exploration: randomly sample actions
                actions = random.sample(available_actions, min(num_actions, len(available_actions)))
                log_probs = torch.log(torch.tensor([1.0 / len(actions)] * len(actions)).to(device))
                entropy = 0
            else:
                # Exploitation: sample from the policy
                masked_probs = masked_probs / masked_probs.sum()
                m = Categorical(masked_probs)
                if num_actions == 1:
                    action = m.sample().item()
                    actions = [action]
                else:
                    actions = m.sample((num_actions,)).flatten().tolist()
                log_probs = m.log_prob(torch.tensor(actions).to(device))
                entropy = m.entropy().sum()

            return actions, log_probs.sum() + self.entropy_coeff * entropy

    def train_step(self, log_prob, reward):
        self.policy_net.train()
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

# -------------------------
# Adversary Class
# -------------------------
class AdversaryWithMemory:
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_hidden_dim, num_classes=3, lr=0.001, entropy_coeff=0.01):
        self.policy_net = GNNPolicyNetworkWithMemory(input_dim, hidden_dim, output_dim, lstm_hidden_dim).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.memory = None
        self.entropy_coeff = entropy_coeff
        self.num_classes = num_classes

    def reset_memory(self):
        self.memory = (torch.zeros(1, 1, self.policy_net.lstm.hidden_size).to(device),
                       torch.zeros(1, 1, self.policy_net.lstm.hidden_size).to(device))

    def select_actions(self, data, available_actions, num_actions, epsilon=0.1):
        self.policy_net.eval()
        with torch.no_grad():
            action_probs, self.memory = self.policy_net(data, self.memory)
            mask = torch.zeros_like(action_probs).to(device)
            mask[0, available_actions] = 1
            masked_probs = action_probs * mask

            if masked_probs.sum() == 0:
                # Fallback to heuristic: select the node with the highest degree
                node_indices = available_actions
                degrees = [graph.degree(node_list[i]) for i in node_indices]
                max_degree_idx = node_indices[np.argmax(degrees)]
                return [max_degree_idx], torch.log(torch.tensor([1.0])).to(device)

            if random.random() < epsilon:
                # Exploration: randomly sample actions
                actions = random.sample(available_actions, min(num_actions, len(available_actions)))
                log_probs = torch.log(torch.tensor([1.0 / len(actions)] * len(actions)).to(device))
                entropy = 0
            else:
                # Exploitation: sample from the policy
                masked_probs = masked_probs / masked_probs.sum()
                m = Categorical(masked_probs)
                if num_actions == 1:
                    action = m.sample().item()
                    actions = [action]
                else:
                    actions = m.sample((num_actions,)).flatten().tolist()
                log_probs = m.log_prob(torch.tensor(actions).to(device))
                entropy = m.entropy().sum()

            return actions, log_probs.sum() + self.entropy_coeff * entropy

    def train_step(self, log_prob, reward):
        self.policy_net.train()
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

# -------------------------
# Reward Computation
# -------------------------
def compute_rewards(controlled_nodes, adversarial_nodes, node_resource_costs, graph,
                   degree_centrality,
                   lambda_resource=0.1, lambda_robustness=0.05,
                   stability_bonus=0.01, instability_penalty=0.05,
                   stability_history={}):
    num_controlled = len([node for node in controlled_nodes if node not in adversarial_nodes])
    total_resource = sum([node_resource_costs[node] for node in controlled_nodes])
    if controlled_nodes:
        subgraph = graph.subgraph(controlled_nodes)
        robustness = nx.average_degree_connectivity(subgraph)
    else:
        robustness = 0

    stability_bonus_total = 0
    instability_penalty_total = 0
    nodes_to_remove = []
    for node, stability in stability_history.items():
        if node in controlled_nodes:
            stability_history[node] += 1
            stability_bonus_total += stability_bonus
        else:
            stability_history[node] -= 1
            if stability_history[node] <= -3:
                instability_penalty_total += instability_penalty * abs(stability_history[node])
                nodes_to_remove.append(node)
    for node in nodes_to_remove:
        del stability_history[node]

    agent_reward = num_controlled - lambda_resource * total_resource + lambda_robustness * robustness + \
                   stability_bonus_total - instability_penalty_total

    adversary_success = len([node for node in adversarial_nodes if node not in controlled_nodes])
    high_value_nodes = [node for node in adversarial_nodes if degree_centrality.get(node, 0) > 0.01]
    blocking_success = len([node for node in high_value_nodes if node not in controlled_nodes])
    adversary_reward = adversary_success + 2 * blocking_success

    return agent_reward, adversary_reward, stability_history

# -------------------------
# Heuristic Fallback
# -------------------------
def heuristic_fallback(agent, adversary, controlled_nodes, adversarial_nodes,
                      graph, subgraph, node_list, feature_matrix, memory,
                      clustering_coeff, stability_history):
    # Agent Action: Control a node with minimum resource cost
    available_agent_actions = [i for i, node in enumerate(node_list)
                               if node not in controlled_nodes and node not in adversarial_nodes]
    if available_agent_actions:
        resource_costs = [feature_matrix[i][0].item() for i in available_agent_actions]  # Assuming resource cost is at index 0
        min_cost_idx = available_agent_actions[np.argmin(resource_costs)]
        agent_selected_node = node_list[min_cost_idx]
        controlled_nodes.add(agent_selected_node)
        agent_reward = 1 - 0.1 * node_resource_costs[agent_selected_node]
        agent.train_step(torch.log(torch.tensor([1.0])).to(device), agent_reward)

    # Adversary Action: Remove a controlled node with the highest degree
    available_adversary_actions = [i for i, node in enumerate(node_list)
                                   if node in controlled_nodes]
    if available_adversary_actions:
        degrees = [graph.degree(node_list[i]) for i in available_adversary_actions]
        max_degree_idx = available_adversary_actions[np.argmax(degrees)]
        adversary_selected_node = node_list[max_degree_idx]
        controlled_nodes.discard(adversary_selected_node)
        adversary_reward = 1
        adversary.train_step(torch.log(torch.tensor([1.0])).to(device), adversary_reward)

# -------------------------
# Control State Initialization
# -------------------------
def initialize_control_states(graph, initial_control, initial_adversarial):
    controlled_nodes = set(random.sample(list(graph.nodes()), initial_control))
    adversarial_nodes = set(random.sample(list(graph.nodes()), initial_adversarial))
    controlled_nodes -= adversarial_nodes  # Ensure no overlap
    return controlled_nodes, adversarial_nodes

# -------------------------
# Detection Probability Computation
# -------------------------
def compute_detection_probability_nn(node, graph, controlled_nodes, adversarial_nodes,
                                    node_resource_costs, detection_predictor, feature_extractor):
    subgraph = graph
    features, node_list = feature_extractor(subgraph, controlled_nodes, adversarial_nodes,
                                           node_resource_costs, {'total_controlled': len(controlled_nodes)},
                                           {}, {})
    node_index = list(subgraph.nodes).index(node)
    node_features = features[node_index].unsqueeze(0)
    with torch.no_grad():
        prob = detection_predictor(node_features)
    return prob.item()

# -------------------------
# Training Loop
# -------------------------
def training_loop(
    graph,
    epochs=100,
    lambda_resource=0.1,
    lambda_robustness=0.05,
    stability_bonus=0.01,
    instability_penalty=0.05,
    num_samples=10000,
    initial_control=10,
    initial_adversarial=5000,
    hidden_dim=128,
    lstm_hidden_dim=256,
    num_heads=3,
    coordinated_actions=5,
    p_add=0.001,
    p_remove=0.001
):
    input_dim = 7  # As defined in extract_features
    output_dim = graph.number_of_nodes()

    agent = AgentWithMemory(input_dim, hidden_dim, output_dim, lstm_hidden_dim, lr=0.001, entropy_coeff=0.01)
    adversary = AdversaryWithMemory(input_dim, hidden_dim, output_dim, lstm_hidden_dim, num_classes=num_heads, lr=0.001, entropy_coeff=0.01)

    adversary_predictor, adversary_optimizer, adversary_scheduler = initialize_adversary_predictor(
        input_dim, hidden_dim, num_classes=num_heads, lr=0.001)

    detection_predictor, detection_optimizer, detection_scheduler = initialize_detection_predictor(
        input_dim + 1, 256, num_classes=2, lr=0.001)  # +1 for temporal feature

    controlled_nodes, adversarial_nodes = initialize_control_states(graph, initial_control, initial_adversarial)

    degree_centrality, betweenness_centrality, eigenvector_centrality, closeness_centrality, katz_centrality = compute_centrality_measures(graph)
    node_resource_costs = initialize_resource_costs(graph, degree_centrality, betweenness_centrality,
                                                  eigenvector_centrality, closeness_centrality, katz_centrality)
    clustering_coeff = nx.clustering(graph)

    stability_history = {}
    agent.reset_memory()
    adversary.reset_memory()

    agent_control_history = []
    adversary_success_history = []
    fallback_usage = 0

    agent_replay_buffer = deque(maxlen=10000)
    adversary_replay_buffer = deque(maxlen=10000)
    detection_replay_buffer = deque(maxlen=10000)

    batch_size = 256

    for epoch in tqdm(range(epochs), desc="Training Progress"):

        if epoch > 0:
            last_agent_success = agent_control_history[-1] if agent_control_history else 0
            last_adversary_success = adversary_success_history[-1] if adversary_success_history else 0
            graph = evolve_graph(graph, degree_centrality, last_agent_success, last_adversary_success, p_add_base=p_add, p_remove_base=p_remove, m=5)
            # Recompute centrality after graph evolution
            degree_centrality, betweenness_centrality, eigenvector_centrality, closeness_centrality, katz_centrality = compute_centrality_measures(graph)

        if epoch % 10 == 0 and epoch != 0:
            degree_centrality, betweenness_centrality, eigenvector_centrality, closeness_centrality, katz_centrality = compute_centrality_measures(graph)
            node_resource_costs = initialize_resource_costs(graph, degree_centrality, betweenness_centrality,
                                                          eigenvector_centrality, closeness_centrality, katz_centrality)
            clustering_coeff = nx.clustering(graph)

        # Update resource costs with fluctuations
        for node in graph.nodes:
            fluctuation = random.uniform(-0.05, 0.05)
            node_resource_costs[node] = max(0.1, node_resource_costs[node] + fluctuation)

        # Sample a subgraph based on community detection
        partition = community.best_partition(graph, resolution=1.0)
        communities = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        community_sizes = np.array([len(members) for members in communities.values()])
        community_probs = community_sizes / community_sizes.sum()
        sampled_communities = np.random.choice(list(communities.keys()), size=num_samples, p=community_probs, replace=True)
        sampled_nodes = set()
        for comm in sampled_communities:
            sampled_nodes.update(communities[comm])
        subgraph = graph.subgraph(sampled_nodes).copy()

        global_state = {'total_controlled': len(controlled_nodes)}
        feature_matrix, node_list = extract_features(subgraph, controlled_nodes, adversarial_nodes,
                                                     node_resource_costs, global_state, stability_history,
                                                     clustering_coeff)
        edge_index = torch.tensor(list(subgraph.edges()), dtype=torch.long).t().contiguous()

        if edge_index.numel() == 0:
            fallback_usage +=1
            heuristic_fallback(agent, adversary, controlled_nodes, adversarial_nodes, graph,
                              subgraph, node_list, feature_matrix, agent.memory, clustering_coeff, stability_history)
            logging.info(f"Epoch {epoch}: Fallback used due to empty subgraph.")
            continue

        data = Data(x=feature_matrix, edge_index=edge_index).to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)
        global_controlled = len(controlled_nodes)
        global_feature = torch.full((data.num_nodes, 1), global_controlled).to(device)
        data.x = torch.cat((data.x, global_feature), dim=1)

        memory_feature = torch.zeros(data.num_nodes, 1).to(device)
        for node in node_list:
            memory_feature[list(subgraph.nodes).index(node)] = stability_history.get(node, 0)
        data.x = torch.cat((data.x, memory_feature), dim=1)

        available_agent_actions = [i for i, node in enumerate(node_list)
                                   if node not in controlled_nodes and node not in adversarial_nodes]
        available_adversary_actions = [i for i, node in enumerate(node_list)
                                       if node in controlled_nodes]

        if not available_agent_actions or not available_adversary_actions:
            fallback_usage +=1
            heuristic_fallback(agent, adversary, controlled_nodes, adversarial_nodes, graph,
                              subgraph, node_list, feature_matrix, agent.memory, clustering_coeff, stability_history)
            logging.info(f"Epoch {epoch}: Fallback used due to no available actions.")
            continue

        # Agent selects actions
        agent_actions, agent_log_prob = agent.select_actions(data, available_agent_actions,
                                                             coordinated_actions, epsilon=0.1)
        agent_selected_nodes = [node_list[idx] for idx in agent_actions]
        controlled_nodes.update(agent_selected_nodes)

        # Adversary selects actions
        adversary_actions, adversary_log_prob = adversary.select_actions(data, available_adversary_actions,
                                                                         coordinated_actions, epsilon=0.1)
        adversary_selected_nodes = [node_list[idx] for idx in adversary_actions]
        for node in adversary_selected_nodes:
            controlled_nodes.discard(node)

        # Compute rewards
        agent_reward, adversary_reward, stability_history = compute_rewards(
            controlled_nodes,
            adversarial_nodes,
            node_resource_costs,
            graph,
            degree_centrality,
            lambda_resource=lambda_resource,
            lambda_robustness=lambda_robustness,
            stability_bonus=stability_bonus,
            instability_penalty=instability_penalty,
            stability_history=stability_history
        )

        # Train agent and adversary
        agent_loss = agent.train_step(agent_log_prob, agent_reward)
        adversary_loss = adversary.train_step(adversary_log_prob, adversary_reward)

        # Add to replay buffers
        agent_replay_buffer.append((agent_log_prob, agent_reward))
        adversary_replay_buffer.append((adversary_log_prob, adversary_reward))

        # Train adversary predictor if buffer is sufficient
        if len(adversary_replay_buffer) >= batch_size:
            batch = random.sample(adversary_replay_buffer, batch_size)
            batch_log_probs, batch_rewards = zip(*batch)
            for log_prob, reward in zip(batch_log_probs, batch_rewards):
                adversary.train_step(log_prob, reward)

        # Train adversary action predictor
        adversary_action_labels = torch.zeros((len(adversary_selected_nodes),)).long().to(device)
        for idx, node in enumerate(adversary_selected_nodes):
            degree = graph.degree(node)
            if degree > 10:
                class_label = 2  # High degree
            elif degree > 5:
                class_label = 1  # Medium degree
            else:
                class_label = 0  # Low degree
            adversary_action_labels[idx] = class_label

        if len(adversary_selected_nodes) > 0:
            adversary_selected_features, _ = extract_features(subgraph, controlled_nodes, adversarial_nodes,
                                                               node_resource_costs, global_state, stability_history,
                                                               clustering_coeff)
            adversary_selected_features = adversary_selected_features[[node_list.index(node) for node in adversary_selected_nodes]]
            loss = train_adversary_predictor(adversary_predictor, adversary_optimizer, adversary_scheduler, adversary_selected_features, adversary_action_labels)

        # Train detection predictor
        detection_labels = []
        detection_features = []
        for node in node_list:
            label = 1 if node in adversarial_nodes else 0
            detection_labels.append(label)
            detection_features.append(feature_matrix[list(subgraph.nodes).index(node)])
        detection_features = torch.stack(detection_features).to(device)
        detection_labels = torch.tensor(detection_labels).to(device)
        detection_replay_buffer.append((detection_features, detection_labels))

        if len(detection_replay_buffer) >= batch_size:
            batch = random.sample(detection_replay_buffer, batch_size)
            batch_features, batch_labels = zip(*batch)
            batch_features = torch.cat(batch_features, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            loss = train_detection_predictor(detection_predictor, detection_optimizer, detection_scheduler, batch_features, batch_labels)

        # Compute detection avoidance
        detection_avoidance = 0
        for node in controlled_nodes:
            prob = compute_detection_probability_nn(node, graph, controlled_nodes, adversarial_nodes,
                                                   node_resource_costs, detection_predictor, extract_features)
            if prob < 0.5:
                detection_avoidance +=1

        # Record history
        agent_control_history.append(len([node for node in controlled_nodes if node not in adversarial_nodes]))
        adversary_success_history.append(adversary_reward)
        logging.info(f"Epoch {epoch}: Agent Reward: {agent_reward}, Adversary Reward: {adversary_reward}, "
                     f"Fallback Usage: {fallback_usage}, Detection Avoidance: {detection_avoidance}")
        logging.info(f"Epoch {epoch}: Controlled Nodes: {len(controlled_nodes)}, Adversary Successes: {adversary_reward}")

        # Adjust coordinated actions over epochs
        if epoch < epochs * 0.5:
            coordinated_actions = max(1, coordinated_actions - 1)
        else:
            coordinated_actions = min(coordinated_actions + 1, 10)

    # Final Logging
    avg_control = np.mean(agent_control_history) if agent_control_history else 0
    max_control = np.max(agent_control_history) if agent_control_history else 0
    min_control = np.min(agent_control_history) if agent_control_history else 0
    total_adversary_success = sum(adversary_success_history) if adversary_success_history else 0
    logging.info("Training completed.")
    logging.info(f"Average Controlled Nodes by Agent: {avg_control}")
    logging.info(f"Max Controlled Nodes in an Epoch by Agent: {max_control}")
    logging.info(f"Min Controlled Nodes in an Epoch by Agent: {min_control}")
    logging.info(f"Total Adversary Successes: {total_adversary_success}")
    print("\nTraining completed. Check training.log for detailed metrics.")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    graph = initialize_graph()
    training_loop(
        graph=graph,
        epochs=100,
        lambda_resource=0.1,
        lambda_robustness=0.05,
        stability_bonus=0.01,
        instability_penalty=0.05,
        num_samples=10000,
        initial_control=10,
        initial_adversarial=5000,
        hidden_dim=128,
        lstm_hidden_dim=256,
        num_heads=3,
        coordinated_actions=5,
        p_add=0.001,
        p_remove=0.001
    )
