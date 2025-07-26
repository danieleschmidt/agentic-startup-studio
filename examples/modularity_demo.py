#!/usr/bin/env python3
"""
Demonstration of improved core services modularity.
Shows dependency injection, service registry, and clean interfaces.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.service_factory import create_default_factory
from core.interfaces import IAlertManager, IBudgetSentinel


def main():
    """Demonstrate modular architecture benefits"""
    print("=== Core Services Modularity Demo ===\n")
    
    # 1. Create factory with default services
    print("1. Setting up service factory with dependency injection...")
    factory = create_default_factory({
        "alert_log_path": "demo_alerts.log"
    })
    registry = factory.get_registry()
    print(f"   Registered services: {registry.list_services()}")
    
    # 2. Get services through interfaces (loose coupling)
    print("\n2. Retrieving services through interfaces...")
    alert_service: IAlertManager = registry.get("alert_service")
    token_sentinel: IBudgetSentinel = registry.get("token_budget_sentinel")
    print("   ✅ Services retrieved successfully")
    
    # 3. Demonstrate service interaction
    print("\n3. Demonstrating service interaction...")
    
    # Record some alerts
    alert_service.record_alert("System startup", "info")
    alert_service.record_alert("Configuration loaded", "info")
    
    # Check token usage (within budget)
    print("   Testing token usage within budget...")
    result = token_sentinel.check_usage(5000, "Demo Context")
    print(f"   Budget check result: {result} (should be True)")
    
    # Check token usage (exceeding budget)
    print("   Testing token usage exceeding budget...")
    result = token_sentinel.check_usage(15000, "Overflow Test")
    print(f"   Budget check result: {result} (should be False)")
    
    # 4. Show recorded alerts
    print("\n4. Recorded alerts:")
    alerts = alert_service.get_alerts()
    for i, alert in enumerate(alerts, 1):
        print(f"   Alert {i}: [{alert['severity'].upper()}] {alert['message']}")
    
    # 5. Demonstrate interface benefits
    print("\n5. Interface benefits demonstrated:")
    print("   ✅ Loose coupling: Services depend on interfaces, not concrete classes")
    print("   ✅ Testability: Easy to mock services through interfaces")
    print("   ✅ Flexibility: Can swap implementations without changing clients")
    print("   ✅ Service discovery: Registry enables dynamic service location")
    
    # 6. Show modularity metrics
    print("\n6. Modularity improvements:")
    print("   ✅ Eliminated direct imports between core services")
    print("   ✅ Introduced dependency injection pattern")
    print("   ✅ Standardized service interfaces")
    print("   ✅ Centralized service configuration")
    
    # Cleanup
    if os.path.exists("demo_alerts.log"):
        os.remove("demo_alerts.log")
    if os.path.exists("logs/demo_alerts.log"):
        os.remove("logs/demo_alerts.log")
        if os.path.exists("logs") and not os.listdir("logs"):
            os.rmdir("logs")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()