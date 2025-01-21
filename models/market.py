class Market:
    def __init__(self):
        self.supply = 0
        self.demand = 0
        self.price = 0.4  # Initialize price at target
        self.target_price = 0.4
        self.price_adjustment_rate = 0.05  # Reduced from 0.1 for more stability
        self.price_stability_threshold = 0.02  # Reduced for tighter control

    def update_market(self, agents):
        """Update market conditions based on agent activities."""
        self.supply = sum(agent.resources for agent in agents)
        self.demand = sum(agent.strategy for agent in agents)
        
        # Normalize supply and demand to prevent extreme values
        total = self.supply + self.demand
        if total > 0:
            self.supply = self.supply / total
            self.demand = self.demand / total
            
        self.price = self.calculate_price()

    def calculate_price(self):
        """Calculate price based on supply and demand with stabilization mechanism."""
        if self.demand <= 0:
            return self.target_price  # Return target price instead of current price
            
        # Calculate market-driven price
        market_price = (self.supply / self.demand) * self.target_price
        
        # Apply price stabilization mechanism
        price_difference = self.target_price - market_price
        
        # If price is within stability threshold, maintain target price
        if abs(price_difference) <= self.price_stability_threshold:
            return self.target_price
        
        # Gradually adjust price towards target
        adjustment = price_difference * self.price_adjustment_rate
        new_price = self.price + adjustment
        
        # Ensure price doesn't move too far from target
        max_price = self.target_price * 1.2  # Reduced from 1.5
        min_price = self.target_price * 0.8  # Increased from 0.5
        
        return max(min(new_price, max_price), min_price)
