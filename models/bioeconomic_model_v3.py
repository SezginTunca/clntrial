

himport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import List
from market import Market  # Assuming Market is defined in market.py

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import random

class Agent:
    def __init__(self, agent_id: int, resources: float = 1.0, strategy: float = 1.0):
        self.id = agent_id
        self.resources = resources
        self.strategy = strategy
        self.performance = 0.0
        self.adaptation_rate = 0.15
        self.target_price = 0.4
        self.min_resources = 0.2
        self.max_resources = 1.8

def learn(self, historical_performance: List[float]) -> None:
    """Adjust strategy based on historical performance."""
    logging.debug(f"Agent {self.id} learning from historical performance: {historical_performance}")
    if len(historical_performance) > 1:
        recent_performance = historical_performance[-1]
        past_performance = historical_performance[-2]
        if recent_performance < past_performance:
            self.strategy *= 0.95  # Decrease strategy if performance dropped
        else:
            self.strategy *= 1.05  # Increase strategy if performance improved
    self.strategy = max(0.2, min(self.strategy, 1.8))

def update(self, market_price: float, environment_condition: float) -> None:
    logging.debug(f"Agent {self.id} updating with market price: {market_price} and environment condition: {environment_condition}")
    price_difference = self.target_price - market_price
    self.strategy += price_difference * self.adaptation_rate
    self.strategy = max(0.2, min(self.strategy, 1.8))

    resource_adjustment = environment_condition * 0.1
    if market_price > self.target_price:
        resource_adjustment *= 1.2
    else:
        resource_adjustment *= 0.8

    self.resources *= (1.0 + resource_adjustment)
    self.resources = max(self.min_resources, min(self.resources, self.max_resources))
    self.performance = self.resources * market_price

class Environment:
def evaluate_impact(self, agents: List[Agent]) -> float:
    logging.debug(f"Evaluating environmental impact for {len(agents)} agents")
    """Evaluate the environmental impact of all agents."""
    total_resources = sum(agent.resources for agent in agents)
    impact = total_resources / len(agents)
    logging.info("Environmental impact evaluated: {:.2f}".format(impact))
    return impact

def __init__(self):
    self.condition = 1.0

def fluctuate(self) -> None:
    self.condition += np.random.normal(0, 0.1)

class BioeconomicModel:
    def __init__(self, num_agents: int):
        self.data: pd.DataFrame | None = None
        self.economic_factors: pd.DataFrame | None = None
        self.sustainability_metrics: pd.Series | None = None
        self.optimization_results: float | None = None
        self.additional_metrics: pd.DataFrame | None = None

        self.agents: List[Agent] = [Agent(i, np.random.rand(), np.random.rand()) for i in range(num_agents)]
        self.market: Market = Market()
        self.environment: Environment = Environment()

        self.market_prices: List[float] = []
        self.environment_conditions: List[float] = []
        self.agent_performances: List[List[float]] = [[] for _ in range(num_agents)]

def generate_hypothetical_data(self, num_samples: int) -> None:
    logging.debug(f"Generating hypothetical data with {num_samples} samples")
    """Generate hypothetical datasets for economic factors and environmental conditions."""
    np.random.seed(42)  # For reproducibility
    self.data = pd.DataFrame({
        'factor1': np.random.normal(loc=5, scale=2, size=num_samples),  # Normal distribution
        'factor2': np.random.uniform(low=1, high=10, size=num_samples),  # Uniform distribution
        'raw': np.random.exponential(scale=50, size=num_samples),  # Exponential distribution
        'metric1': np.random.beta(a=2, b=5, size=num_samples) * 100,  # Beta distribution
        'metric2': np.random.poisson(lam=20, size=num_samples)  # Poisson distribution
    })
    logging.info("Hypothetical datasets generated with advanced distributions.")

def simulate(self, steps: int = 10) -> None:
    logging.debug(f"Starting simulation for {steps} steps")
    logging.info(f"Starting simulation for {steps} steps.")
    for step in range(steps):
        self.environment.fluctuate()
        for agent in self.agents:
            agent.update(self.market.price, self.environment.condition)

        self.market.update_market(self.agents)
        self.record_state()
        # Agents learn from their performance
        for agent in self.agents:
            agent.learn(self.agent_performances[agent.id])
        # Evaluate environmental impact
        self.environment.evaluate_impact(self.agents)
    logging.info("Simulation complete.")

def record_state(self) -> None:
    logging.debug(f"Recording state: market price {self.market.price}, environment condition {self.environment.condition}")
    self.market_prices.append(self.market.price)
    self.environment_conditions.append(self.environment.condition)
    for i, agent in enumerate(self.agents):
        self.agent_performances[i].append(agent.performance)

def process_data(self) -> None:
    logging.debug("Processing data")
    if self.data is not None:
        self.data['processed'] = self.data['raw'] * 1.5

def analyze_economics(self) -> None:
    logging.debug("Analyzing economics")
    if self.economic_factors is not None:
        self.economic_factors['analysis'] = self.economic_factors.mean(axis=1)

def evaluate_sustainability(self) -> None:
    logging.debug("Evaluating sustainability")
    if self.data is not None:
        self.sustainability_metrics = self.data['processed'] / self.data['factor1']
        if self.additional_metrics is not None:
            self.sustainability_metrics += self.additional_metrics.sum(axis=1)

def optimize(self) -> None:
    logging.debug("Optimizing")
    if self.sustainability_metrics is not None:
        self.optimization_results = self.sustainability_metrics.max()

def calculate_performance_metrics(self) -> None:
    logging.debug("Calculating performance metrics")
    """Calculate additional performance metrics for agents."""
    self.performance_metrics = {
        'mean': [],
        'variance': [],
        'std_dev': [],
        'confidence_interval': []
    }

    for performances in self.agent_performances:
        self.performance_metrics['mean'].append(np.mean(performances))
        self.performance_metrics['variance'].append(np.var(performances))
        self.performance_metrics['std_dev'].append(np.std(performances))
        self.performance_metrics['confidence_interval'].append(
            1.96 * (np.std(performances) / np.sqrt(len(performances)))
        )
    logging.info("Performance metrics calculated.")

def generate_report(self) -> None:
    logging.debug("Generating report")
    """Generate a report of the simulation results and save to CSV."""
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)

    summary_data = {
        'Time Step': list(range(len(self.market_prices))),
        'Market Price': self.market_prices,
        'Environmental Condition': self.environment_conditions
    }

    for i, performances in enumerate(self.agent_performances):
        summary_data[f'Agent {i} Performance'] = performances

    df = pd.DataFrame(summary_data)
    csv_path = f'{output_dir}/simulation_results_v3.csv'
    df.to_csv(csv_path, index=False)

    # Add performance metrics to the report
    metrics_df = pd.DataFrame(self.performance_metrics)
    metrics_csv_path = f'{output_dir}/performance_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)

    logging.info("Simulation Summary Statistics:")
    logging.info("-" * 30)
    logging.info(f"Average Market Price: {np.mean(self.market_prices):.2f}")
    logging.info(f"Average Environmental Condition: {np.mean(self.environment_conditions):.2f}")

    for i, performances in enumerate(self.agent_performances):
        logging.info(f"Agent {i} Average Performance: {np.mean(performances):.2f}")

def plot_agent_performance(self, output_dir: str) -> None:
    logging.debug(f"Plotting agent performance to {output_dir}")
    """Plot individual agent performances over time."""
    logging.info("Generating plots for individual agent performances.")
    plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
    for i, performances in enumerate(self.agent_performances):
        plt.plot(performances, label=f'Agent {i} Performance', linewidth=2)  # Thicker lines for visibility
    plt.title('Agent Performance Over Time', fontsize=16)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Performance', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)  # Adjust legend position
    plt.grid(True)  # Add grid lines
    plt.savefig(f'{output_dir}/agent_performance.png')
    plt.close()
    logging.info("Generated plots for individual agent performances.")

def plot_correlation_matrix(self, output_dir: str) -> None:
    logging.debug(f"Plotting correlation matrix to {output_dir}")
    """Plot correlation matrix for economic factors."""
    logging.info("Generating correlation matrix plot.")
    if self.data is not None:
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.title('Correlation Matrix of Economic Factors')
        plt.savefig(f'{output_dir}/correlation_matrix.png')
        plt.close()
        logging.info("Generated correlation matrix plot.")

def plot_histograms(self, output_dir: str) -> None:
    logging.debug(f"Plotting histograms to {output_dir}")
    """Plot histograms for economic factors."""
    logging.info("Generating histograms for economic factors.")
    for column in self.data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[column], bins=30, alpha=0.7)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/histogram_{column}.png')
        plt.close()
    logging.info("Generated histograms for economic factors.")

def plot_boxplots(self, output_dir: str) -> None:
    logging.debug(f"Plotting boxplots to {output_dir}")
    """Plot boxplots for economic factors."""
    logging.info("Generating boxplots for economic factors.")
    for column in self.data.columns:
        plt.figure(figsize=(10, 6))
        plt.boxplot(self.data[column])
        plt.title(f'Boxplot of {column}')
        plt.ylabel(column)
        plt.savefig(f'{output_dir}/boxplot_{column}.png')
        plt.close()
    logging.info("Generated boxplots for economic factors.")

def plot_additional_charts(self, output_dir: str) -> None:
    logging.debug(f"Plotting additional charts to {output_dir}")
    """Generate additional visualisation charts."""
    logging.info("Generating additional visualisation charts.")
    if self.data is not None:
        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['factor1'], self.data['metric1'], alpha=0.7)
        plt.title('Scatter Plot of Factor1 vs Metric1')
        plt.xlabel('Factor1')
        plt.ylabel('Metric1')
        plt.savefig(f'{output_dir}/scatter_factor1_metric1.png')
        plt.close()

        # Line plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['factor2'], self.data['metric2'], alpha=0.7)
        plt.title('Line Plot of Factor2 vs Metric2')
        plt.xlabel('Factor2')
        plt.ylabel('Metric2')
        plt.savefig(f'{output_dir}/line_factor2_metric2.png')
        plt.close()

        logging.info("Generated additional visualisation charts.")

    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)

    # Advanced Time Series Plot
    plt.figure(figsize=(12, 6))
    for i, performances in enumerate(self.agent_performances):
        plt.plot(performances, label=f'Agent {i} Performance')
    plt.title('Agent Performance Over Time with Trend Lines')
    plt.xlabel('Time Step')
    plt.ylabel('Performance')
    plt.legend()
    plt.savefig(f'{output_dir}/advanced_agent_performance.png')
    plt.close()

    # Heatmap for Correlation Matrix
    if self.data is not None:
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Advanced Correlation Matrix Heatmap')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.savefig(os.path.join(output_dir, 'advanced_correlation_matrix.png'))
        plt.close()

    logging.info("Generated advanced plots.")
    logging.info("Generated all required plots.")

    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)

    # Generate multiple plots
    self.plot_agent_performance(output_dir)
    self.plot_correlation_matrix(output_dir)
    self.plot_histograms(output_dir)
    self.plot_boxplots(output_dir)

    logging.info("Generated all required plots.")

if __name__ == "__main__":
    logging.debug("Starting main execution")
    np.random.seed(42)  # For reproducible simulation
    model = BioeconomicModel(num_agents=10)
    model.generate_hypothetical_data(num_samples=100)  # Generate hypothetical data
    model.simulate(steps=50)
    model.plot_results()
    model.plot_advanced_results()
    model.generate_report()
```
