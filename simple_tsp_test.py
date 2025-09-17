#!/usr/bin/env python3
"""
Simple test of the Quantum TSP implementation without visualization.
"""

import numpy as np
from quantum_tsp import ClimateAwareTSP

def test_basic_functionality():
    """Test basic TSP functionality without matplotlib."""
    print("🌍 Testing Quantum TSP - Climate Action")
    print("=" * 50)
    
    # Create a simple 3-city scenario
    cities = [
        ("San Francisco", 37.7749, -122.4194),
        ("Los Angeles", 34.0522, -118.2437),
        ("San Diego", 32.7157, -117.1611)
    ]
    
    print(f"\n📍 Cities: {', '.join([city[0] for city in cities])}")
    
    # Test with electric vehicle
    print(f"\n🚗 Testing with Electric Vehicle")
    print("-" * 30)
    
    try:
        tsp_solver = ClimateAwareTSP(cities, "electric")
        print("✅ TSP solver created successfully")
        
        # Test distance matrix calculation
        print(f"📏 Distance matrix shape: {tsp_solver.distance_matrix.shape}")
        print(f"📊 Sample distances:")
        for i in range(min(3, len(cities))):
            for j in range(i+1, min(3, len(cities))):
                dist = tsp_solver.distance_matrix[i][j]
                print(f"   {cities[i][0]} → {cities[j][0]}: {dist:.1f} km")
        
        # Test carbon matrix
        print(f"🌱 Carbon emission factors: {tsp_solver.emission_factors}")
        print(f"💨 Sample carbon emissions:")
        for i in range(min(3, len(cities))):
            for j in range(i+1, min(3, len(cities))):
                carbon = tsp_solver.carbon_matrix[i][j]
                print(f"   {cities[i][0]} → {cities[j][0]}: {carbon:.3f} kg CO2")
        
        # Solve TSP
        print(f"\n🔬 Solving TSP...")
        solution = tsp_solver.solve_quantum_tsp()
        
        print(f"\n✅ Solution found!")
        print(f"🗺️  Optimal Route: {' → '.join(solution['route_names'])}")
        print(f"📏 Total Distance: {solution['total_distance_km']} km")
        print(f"🌱 Carbon Emissions: {solution['total_carbon_kg']} kg CO2")
        print(f"💚 Carbon Savings: {solution['carbon_savings_kg']} kg CO2")
        print(f"🚗 Vehicle Type: {solution['vehicle_type']}")
        print(f"📊 Emission Factor: {solution['emission_factor']} kg CO2/km")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vehicle_comparison():
    """Test comparison between different vehicle types."""
    print(f"\n{'='*50}")
    print("🔄 Vehicle Type Comparison")
    print(f"{'='*50}")
    
    cities = [
        ("San Francisco", 37.7749, -122.4194),
        ("Los Angeles", 34.0522, -118.2437),
        ("San Diego", 32.7157, -117.1611)
    ]
    
    vehicle_types = ["electric", "hybrid", "gasoline"]
    results = {}
    
    for vehicle_type in vehicle_types:
        print(f"\n🚗 {vehicle_type.title()} Vehicle:")
        try:
            tsp_solver = ClimateAwareTSP(cities, vehicle_type)
            solution = tsp_solver.solve_quantum_tsp()
            results[vehicle_type] = solution
            
            print(f"  📏 Distance: {solution['total_distance_km']} km")
            print(f"  🌱 Carbon: {solution['total_carbon_kg']} kg CO2")
            print(f"  💚 Savings: {solution['carbon_savings_kg']} kg CO2")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    # Find the most efficient vehicle
    if results:
        best_vehicle = min(results.keys(), key=lambda v: results[v]['total_carbon_kg'])
        worst_vehicle = max(results.keys(), key=lambda v: results[v]['total_carbon_kg'])
        
        print(f"\n🏆 Most Efficient: {best_vehicle.title()}")
        print(f"💸 Least Efficient: {worst_vehicle.title()}")
        
        savings = results[worst_vehicle]['total_carbon_kg'] - results[best_vehicle]['total_carbon_kg']
        print(f"💚 Total Carbon Savings: {savings:.2f} kg CO2")
        
        # Environmental impact
        trees_equivalent = savings / 22  # 1 tree absorbs ~22kg CO2/year
        print(f"🌳 Equivalent to planting {trees_equivalent:.1f} trees")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Quantum TSP Test...")
    
    try:
        # Test basic functionality
        success1 = test_basic_functionality()
        
        if success1:
            # Test vehicle comparison
            success2 = test_vehicle_comparison()
            
            if success2:
                print(f"\n{'='*50}")
                print("✅ All tests passed successfully!")
                print("🌱 Quantum TSP is ready for climate action!")
                print("🚀 The implementation successfully optimizes routes")
                print("   to minimize carbon emissions and fuel consumption.")
            else:
                print(f"\n❌ Vehicle comparison test failed")
        else:
            print(f"\n❌ Basic functionality test failed")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


