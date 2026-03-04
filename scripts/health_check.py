"""
ProofCoach API Health Check

Tests the deployed API for basic functionality.
"""

import sys
import time

import httpx

BASE_URL = "http://localhost:8080"


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}", end="")
    if detail:
        print(f": {detail}", end="")
    print()
    if not ok:
        sys.exit(1)


def main():
    print("ProofCoach API Health Check")
    print("=" * 40)

    # 1. Health endpoint
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        check("Health endpoint", r.status_code == 200, r.text[:50])
    except Exception as e:
        check("Health endpoint", False, str(e))

    # 2. Tutor endpoint
    try:
        payload = {
            "problem": "What is the sum of the first 100 positive integers?",
            "student_work": "I think I need to add them all up one by one.",
            "session_id": "health_test_001",
        }
        r = httpx.post(f"{BASE_URL}/v1/tutor", json=payload, timeout=30)
        data = r.json()
        ok = r.status_code == 200 and "question" in data
        check("POST /v1/tutor", ok, data.get("question", "")[:80])
    except Exception as e:
        check("POST /v1/tutor", False, str(e))

    # 3. Verify endpoint
    try:
        payload = {
            "claim": "n^2 - 1 = (n-1)(n+1) for all integers n",
        }
        r = httpx.post(f"{BASE_URL}/v1/verify", json=payload, timeout=30)
        data = r.json()
        ok = r.status_code == 200 and "verified" in data
        check("POST /v1/verify", ok, f"verified={data.get('verified')}")
    except Exception as e:
        check("POST /v1/verify", False, str(e))

    # 4. Diagnose endpoint
    try:
        payload = {
            "problem": "Prove that the sum of two odd numbers is even.",
            "student_work": "Let the two odd numbers be 2k and 2m. Then 2k + 2m = 2(k+m) which is even.",
            "student_answer": "2(k+m)",
        }
        r = httpx.post(f"{BASE_URL}/v1/diagnose", json=payload, timeout=30)
        data = r.json()
        ok = r.status_code == 200
        check("POST /v1/diagnose", ok, data.get("misconception", data.get("correct", ""))[:80])
    except Exception as e:
        check("POST /v1/diagnose", False, str(e))

    # 5. Sequence endpoint
    try:
        payload = {
            "student_id": "health_test_student",
            "session_history": [],
        }
        r = httpx.post(f"{BASE_URL}/v1/sequence", json=payload, timeout=10)
        ok = r.status_code == 200
        check("POST /v1/sequence", ok)
    except Exception as e:
        check("POST /v1/sequence", False, str(e))

    print("=" * 40)
    print("All checks passed.")


if __name__ == "__main__":
    main()
