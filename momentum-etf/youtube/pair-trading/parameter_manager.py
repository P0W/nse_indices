"""
Configuration Manager for Strategy Parameters

This module helps manage and update strategy parameters based on experiment results.
"""

import json
import os
from datetime import datetime
from tabulate import tabulate


class ParameterManager:
    """
    Manage strategy parameters and apply optimal configurations
    """

    def __init__(self):
        self.config_file = "strategy_config.json"
        self.backup_dir = "config_backups"
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def get_current_parameters(self):
        """Get current strategy parameters"""
        from adaptive_momentum import STRATEGY_PARAMS

        return STRATEGY_PARAMS.copy()

    def save_current_config(self, description="Current configuration"):
        """Save current configuration to file"""
        current_params = self.get_current_parameters()

        config = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "parameters": current_params,
        }

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Also create a backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"config_backup_{timestamp}.json")
        with open(backup_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Configuration saved to {self.config_file}")
        print(f"📁 Backup created at {backup_file}")

    def load_config(self, config_file=None):
        """Load configuration from file"""
        file_to_load = config_file or self.config_file

        if not os.path.exists(file_to_load):
            print(f"❌ Configuration file {file_to_load} not found")
            return None

        with open(file_to_load, "r") as f:
            config = json.load(f)

        return config

    def apply_parameters(self, new_parameters, description="Applied from experiments"):
        """
        Apply new parameters to the strategy

        Args:
            new_parameters: Dictionary of new parameters
            description: Description of the change
        """
        # Backup current configuration first
        self.save_current_config(f"Before: {description}")

        # Update the adaptive_momentum module
        import adaptive_momentum

        print("🔄 Applying new parameters...")
        print("\n📊 PARAMETER CHANGES")
        print("=" * 50)

        changes_data = [["Parameter", "Old Value", "New Value", "Change"]]

        for param, new_value in new_parameters.items():
            if param in adaptive_momentum.STRATEGY_PARAMS:
                old_value = adaptive_momentum.STRATEGY_PARAMS[param]
                adaptive_momentum.STRATEGY_PARAMS[param] = new_value

                change_indicator = "➡️" if old_value != new_value else "="
                changes_data.append([param, old_value, new_value, change_indicator])

        print(tabulate(changes_data, headers="firstrow", tablefmt="grid"))

        # Save the new configuration
        self.save_current_config(description)

        print(f"\n✅ Parameters updated successfully!")
        print("💡 Run the main strategy to test the new parameters")

    def compare_configurations(self, config1_file, config2_file=None):
        """Compare two configurations"""
        config1 = self.load_config(config1_file)
        if config2_file:
            config2 = self.load_config(config2_file)
        else:
            config2 = {
                "parameters": self.get_current_parameters(),
                "description": "Current configuration",
            }

        if not config1 or not config2:
            return

        print("\n📊 CONFIGURATION COMPARISON")
        print("=" * 60)
        print(f"Config 1: {config1.get('description', 'Unknown')}")
        print(f"Config 2: {config2.get('description', 'Unknown')}")
        print()

        params1 = config1["parameters"]
        params2 = config2["parameters"]

        comparison_data = [["Parameter", "Config 1", "Config 2", "Different"]]

        all_params = set(params1.keys()) | set(params2.keys())
        for param in sorted(all_params):
            val1 = params1.get(param, "N/A")
            val2 = params2.get(param, "N/A")
            different = "✓" if val1 != val2 else ""
            comparison_data.append([param, val1, val2, different])

        print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))

    def list_backups(self):
        """List all configuration backups"""
        if not os.path.exists(self.backup_dir):
            print("📁 No backup directory found")
            return

        backups = [f for f in os.listdir(self.backup_dir) if f.endswith(".json")]
        backups.sort(reverse=True)  # Most recent first

        if not backups:
            print("📁 No configuration backups found")
            return

        print("\n📁 CONFIGURATION BACKUPS")
        print("=" * 50)

        backup_data = [["#", "Filename", "Date", "Description"]]

        for i, backup in enumerate(backups[:10], 1):  # Show last 10
            backup_path = os.path.join(self.backup_dir, backup)
            config = self.load_config(backup_path)

            if config:
                timestamp = config.get("timestamp", "Unknown")
                description = config.get("description", "No description")
                date_str = timestamp.split("T")[0] if "T" in timestamp else timestamp

                backup_data.append([i, backup, date_str, description])

        print(tabulate(backup_data, headers="firstrow", tablefmt="grid"))

    def restore_backup(self, backup_filename):
        """Restore configuration from backup"""
        backup_path = os.path.join(self.backup_dir, backup_filename)

        if not os.path.exists(backup_path):
            print(f"❌ Backup file {backup_filename} not found")
            return

        config = self.load_config(backup_path)
        if config and "parameters" in config:
            self.apply_parameters(
                config["parameters"], f"Restored from backup: {backup_filename}"
            )
        else:
            print("❌ Invalid backup file format")


def apply_optimal_parameters(experiments_result_file):
    """
    Apply optimal parameters from experiments result file

    Args:
        experiments_result_file: Path to the experiments JSON result file
    """
    if not os.path.exists(experiments_result_file):
        print(f"❌ Experiments result file {experiments_result_file} not found")
        return

    with open(experiments_result_file, "r") as f:
        results = json.load(f)

    if not results:
        print("❌ No results found in the file")
        return

    # Find the best result
    best_result = max(results, key=lambda x: x.get("composite_score", -float("inf")))
    optimal_params = best_result["params"]

    print("🎯 Found optimal parameters from experiments:")
    print(f"   Composite Score: {best_result['composite_score']:.2f}")
    print(f"   Total Return: {best_result['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {best_result['max_drawdown']:.2f}%")

    # Apply the parameters
    manager = ParameterManager()
    manager.apply_parameters(optimal_params, "Optimal from experiments")


def interactive_parameter_manager():
    """Interactive parameter management interface"""
    manager = ParameterManager()

    while True:
        print("\n🛠️ PARAMETER MANAGER")
        print("=" * 30)
        print("1. View current parameters")
        print("2. Save current configuration")
        print("3. Load configuration")
        print("4. List backups")
        print("5. Restore from backup")
        print("6. Compare configurations")
        print("7. Apply optimal from experiments")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ").strip()

        if choice == "1":
            current = manager.get_current_parameters()
            print("\n📊 CURRENT PARAMETERS")
            print("=" * 30)
            param_data = [["Parameter", "Value"]]
            for param, value in current.items():
                param_data.append([param, value])
            print(tabulate(param_data, headers="firstrow", tablefmt="grid"))

        elif choice == "2":
            description = input("Enter description (optional): ").strip()
            if not description:
                description = "Manual save"
            manager.save_current_config(description)

        elif choice == "3":
            config_file = input(
                "Enter config file path (or press Enter for default): "
            ).strip()
            config = manager.load_config(config_file if config_file else None)
            if config:
                print(
                    f"✅ Loaded configuration: {config.get('description', 'No description')}"
                )
                apply = input("Apply these parameters? (y/n): ").strip().lower()
                if apply == "y":
                    manager.apply_parameters(config["parameters"], "Loaded from file")

        elif choice == "4":
            manager.list_backups()

        elif choice == "5":
            manager.list_backups()
            backup_name = input("\nEnter backup filename: ").strip()
            if backup_name:
                manager.restore_backup(backup_name)

        elif choice == "6":
            print("Compare configurations:")
            file1 = input("Enter first config file: ").strip()
            file2 = input(
                "Enter second config file (or press Enter for current): "
            ).strip()
            manager.compare_configurations(file1, file2 if file2 else None)

        elif choice == "7":
            experiment_file = input("Enter experiments result JSON file path: ").strip()
            if experiment_file:
                apply_optimal_parameters(experiment_file)

        elif choice == "8":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    interactive_parameter_manager()
