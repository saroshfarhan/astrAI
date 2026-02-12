"""Quick Spectral Service Integration Test

This script tests the Spectral Service without needing the full agent system.
Tests:
1. Service health
2. Endpoint functionality
3. Response format

Run this AFTER starting the Spectral Service.
"""

import sys
import requests
from pathlib import Path


def test_service_health():
    """Test 1: Check if Spectral Service is running."""
    print("\n" + "="*60)
    print("TEST 1: Spectral Service Health Check")
    print("="*60)

    service_url = "http://localhost:8001"
    try:
        response = requests.get(service_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            print("[OK] Service is RUNNING")
            print(f"   URL: {service_url}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
            return True
        else:
            print(f"[FAIL] Service returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to service at {service_url}")
        print("   Make sure to run: start_spectral_service.bat")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_service_endpoint():
    """Test 2: Test /analyze_spectrum endpoint directly."""
    print("\n" + "="*60)
    print("TEST 2: Endpoint Test (Direct API Call)")
    print("="*60)

    test_file = Path("Spectral Service/data/real/mars_uv.pkl")

    if not test_file.exists():
        print(f"[WARN]  Test file not found: {test_file}")
        print("   Skipping direct endpoint test")
        return None

    endpoint = "http://localhost:8001/analyze_spectrum"

    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/octet-stream')}
            params = {'top_k': 5}

            response = requests.post(endpoint, files=files, params=params, timeout=30)

        response.raise_for_status()
        result = response.json()

        print("[OK] Endpoint responded successfully")
        print(f"   Domain used: {result.get('domain_used', 'N/A')}")
        print(f"   Predictions: {len(result.get('predictions', []))}")

        # Print first 3 predictions
        predictions = result.get('predictions', [])[:3]
        for i, pred in enumerate(predictions, 1):
            elem = pred.get('element', '?')
            prob = pred.get('probability', 0)
            print(f"   {i}. {elem}: {prob:.3f}")

        return True

    except requests.exceptions.HTTPError as e:
        print(f"[FAIL] HTTP Error: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_format_compatibility():
    """Test 3: Verify response format matches agent expectations."""
    print("\n" + "="*60)
    print("TEST 3: Format Compatibility Check")
    print("="*60)

    # Expected fields in Prediction
    required_fields = ["element", "probability", "rationale", "color"]

    test_file = Path("Spectral Service/data/real/mars_uv.pkl")

    if not test_file.exists():
        print(f"[WARN]  Test file not found: {test_file}")
        return None

    endpoint = "http://localhost:8001/analyze_spectrum"

    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/octet-stream')}
            response = requests.post(endpoint, files=files, timeout=30)

        result = response.json()
        predictions = result.get('predictions', [])

        if not predictions:
            print("[WARN]  No predictions to check format")
            return None

        # Check first prediction
        pred = predictions[0]
        missing_fields = [f for f in required_fields if f not in pred]

        if missing_fields:
            print(f"[FAIL] Missing required fields: {missing_fields}")
            print(f"   Prediction keys: {list(pred.keys())}")
            return False

        print("[OK] Response format is compatible")
        print(f"   All required fields present: {required_fields}")

        # Check types
        checks = [
            (isinstance(pred['element'], str), "element is string"),
            (isinstance(pred['probability'], (int, float)), "probability is numeric"),
            (pred['rationale'] is None or isinstance(pred['rationale'], str), "rationale is optional string"),
            (pred['color'] is None or isinstance(pred['color'], str), "color is optional string"),
        ]

        all_passed = True
        for check, desc in checks:
            status = "[OK]" if check else "[FAIL]"
            print(f"   {status} {desc}")
            if not check:
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_multiple_files():
    """Test 4: Test with multiple file types."""
    print("\n" + "="*60)
    print("TEST 4: Multiple File Test (UV & IR)")
    print("="*60)

    test_files = [
        Path("Spectral Service/data/real/mars_uv.pkl"),
        Path("Spectral Service/data/real/saturn_ir.pkl"),
    ]

    endpoint = "http://localhost:8001/analyze_spectrum"
    results = []

    for test_file in test_files:
        if not test_file.exists():
            print(f"[WARN]  Skipping {test_file.name} (not found)")
            continue

        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'application/octet-stream')}
                response = requests.post(endpoint, files=files, timeout=30)

            response.raise_for_status()
            result = response.json()

            domain = result.get('domain_used', '?')
            pred_count = len(result.get('predictions', []))
            print(f"[OK] {test_file.name}: Domain={domain}, Predictions={pred_count}")
            results.append(True)

        except Exception as e:
            print(f"[FAIL] {test_file.name}: {e}")
            results.append(False)

    return all(results) if results else None


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("   SPECTRAL SERVICE INTEGRATION TEST")
    print("=" * 60)

    results = {
        "Service Health": test_service_health(),
        "Endpoint Test": test_service_endpoint(),
        "Format Compatibility": test_format_compatibility(),
        "Multiple Files": test_multiple_files(),
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"

        print(f"{status:8} {test_name}")

    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])

    print("="*60)
    if passed == total and total > 0:
        print("SUCCESS! ALL TESTS PASSED! Integration is working.")
        print("\nNext Steps:")
        print("1. The Spectral Service is running correctly")
        print("2. Agent patch has been applied")
        print("3. You can now use the full system:")
        print("   streamlit run app.py")
    elif passed > 0:
        print(f"WARNING: {passed}/{total} tests passed. Check failures above.")
    else:
        print("FAILED: INTEGRATION FAILED. Check that service is running.")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
