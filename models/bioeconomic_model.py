import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from market import Market
import os

class Agent:
    def __init__(self, id, resources=1.0, strategy=1.0):
        self.id = id
        self.resources = resources
        self.strategy = strategy
        self.performance = 0
        self.adaptation_rate = 0.15  # Increased from 0.1
        self.target_price = 0.4
        self.min_resources = 0.2
        self.max_resources = 1.8

    def update(self, market_price, environment_condition):
        """Update agent's strategy and resources based on market conditions."""
        # Adjust strategy based on market price
        price_difference = self.target_price - market_price
        self.strategy += price_difference * self.adaptation_rate
        
        # More aggressive bounds for strategy
        self.strategy = max(0.2, min(self.strategy, 1.8))
        
        # Update resources based on environmental conditions and current market
        resource_adjustment = environment_condition * 0.1
        if market_price > self.target_price:
            resource_adjustment *= 1.2  # Increase production when prices are high
        else:
            resource_adjustment *= 0.8  # Decrease production when prices are low
            
        self.resources *= (1.0 + resource_adjustment)
        self.resources = max(self.min_resources, min(self.resources, self.max_resources))
        
        # Calculate performance
        self.performance = self.resources * market_price


class Environment:
    def __init__(self):
        self.condition = 1.0

    def fluctuate(self):
        """Introduce random fluctuations to simulate environmental uncertainty."""
        self.condition += np.random.normal(0, 0.1)  # Random fluctuation


class BioeconomicModel:
    def __init__(self, num_agents):
        self.data = None
        self.economic_factors = None
        self.sustainability_metrics = None
        self.optimization_results = None
        self.additional_metrics = None
        self.agents = [Agent(i, np.random.rand(), np.random.rand()) for i in range(num_agents)]
        self.market = Market()
        self.environment = Environment()
        self.market_prices = []
        self.environment_conditions = []
        self.agent_performances = [[] for _ in range(num_agents)]

    def simulate(self, steps=10):
        """Run the simulation of agents, market dynamics, and environmental uncertainty."""
        for _ in range(steps):
            self.environment.fluctuate()
            for agent in self.agents:
                agent.update(self.market.price, self.environment.condition)
            self.market.update_market(self.agents)
            self.record_state()

    def record_state(self):
        """Record the current state for visualization."""
        self.market_prices.append(self.market.price)
        self.environment_conditions.append(self.environment.condition)
        for i, agent in enumerate(self.agents):
            self.agent_performances[i].append(agent.performance)

    def load_data(self, file_path):
        """Load data from a file."""
        self.data = pd.read_csv(file_path)
        # Assume specific columns based on diagram
        self.economic_factors = self.data[['factor1', 'factor2']]
        self.additional_metrics = self.data[['metric1', 'metric2']]

    def process_data(self):
        """Process the data for analysis."""
        # Implement data processing steps based on diagram
        self.data['processed'] = self.data['raw'] * 1.5  # Example transformation

    def analyze_economics(self):
        """Perform economic analysis."""
        # Implement economic analysis based on diagram
        self.economic_factors['analysis'] = self.economic_factors.mean(axis=1)

    def evaluate_sustainability(self):
        """Evaluate sustainability metrics."""
        # Implement sustainability evaluation based on diagram
        self.sustainability_metrics = self.data['processed'] / self.data['factor1']
        self.sustainability_metrics += self.additional_metrics.sum(axis=1)

    def optimize(self):
        """Optimize the model for best outcomes."""
        # Implement optimization logic based on diagram
        self.optimization_results = self.sustainability_metrics.max()

    def generate_report(self):
        """Generate a report of the results and save to CSV."""
        output_dir = '../output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create summary statistics
        summary_data = {
            'Time Step': range(len(self.market_prices)),
            'Market Price': self.market_prices,
            'Environmental Condition': self.environment_conditions
        }

        # Add agent performances
        for i, performances in enumerate(self.agent_performances):
            summary_data[f'Agent {i} Performance'] = performances

        # Convert to DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(f'{output_dir}/simulation_results.csv', index=False)

        # Print summary statistics
        print("\nSimulation Summary Statistics:")
        print("-" * 30)
        print(f"Average Market Price: {np.mean(self.market_prices):.2f}")
        print(f"Average Environmental Condition: {np.mean(self.environment_conditions):.2f}")
        for i, performances in enumerate(self.agent_performances):
            print(f"Agent {i} Average Performance: {np.mean(performances):.2f}")

    def plot_results(self):
        """Plot the results of the simulation and save them."""
        # Create output directory if it doesn't exist
        output_dir = '../output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Plot Market Price
        plt.figure(figsize=(10, 6))
        plt.plot(self.market_prices, label='Market Price')
        plt.title('Market Price Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'{output_dir}/market_prices.png')
        plt.close()

        # Plot Environmental Condition
        plt.figure(figsize=(10, 6))
        plt.plot(self.environment_conditions, label='Environmental Condition', color='green')
        plt.title('Environmental Condition Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Condition')
        plt.legend()
        plt.savefig(f'{output_dir}/environmental_conditions.png')
        plt.close()

        # Plot Agent Performances
        plt.figure(figsize=(10, 6))
        for i, performances in enumerate(self.agent_performances):
            plt.plot(performances, label=f'Agent {i} Performance')
        plt.title('Agent Performance Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Performance')
        plt.legend()
        plt.savefig(f'{output_dir}/agent_performances.png')
        plt.close()


# Example usage
if __name__ == "__main__":
    model = BioeconomicModel(num_agents=10)
    model.simulate(steps=50)  # Increased steps for better visualization
    model.plot_results()
    model.generate_report()
