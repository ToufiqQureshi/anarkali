"""Build the Anarkali routing hard-suite benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "benchmarks" / "anarkali-routing-v1" / "cases.jsonl"

COMMON = {
    "billing": "Billing, invoices, renewals, refunds, subscriptions, or plan entitlements.",
    "identity": "Authentication, SSO, SCIM, IdP configuration, or user provisioning.",
    "product": "Customer-specific product behavior, workspace configuration, roles, permissions, or feature behavior.",
    "incident": "Widespread service degradation affecting many unrelated customers or regions.",
}

SECURITY = {
    "identity": "Normal authentication, SSO, SCIM, IdP configuration, or user provisioning.",
    "security": "Credential compromise, suspicious access, exposed secrets, or potential account takeover.",
    "product": "Customer-specific product behavior or configuration.",
    "billing": "Billing, invoices, renewals, refunds, subscriptions, or plan entitlements.",
}


def route_case(case_id: str, ticket: str, expected: str, *, criteria: dict[str, str] | None = None,
               tier: str = "enterprise", tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": case_id,
        "state": {"ticket": ticket, "tier": tier},
        "questions": {"route": {"type": "choice", "instructions": "Choose the primary queue.",
                                 "criteria": criteria or COMMON}},
        "expected": {"route": expected},
        "tags": tags or [],
    }


def cases() -> list[dict[str, Any]]:
    rows = [
        route_case("x_scope_single_customer", "The dashboard returns HTTP 500 for every user in one customer workspace. Other customers are healthy and platform monitoring shows normal error rates.", "product", tags=["scope", "seed"]),
        route_case("x_scope_global", "The dashboard returns HTTP 500 for users across 87 customer workspaces. Multiple regions are affected and platform monitoring shows elevated error rates.", "incident", tags=["scope", "seed"]),
        route_case("x_identity_normal_login", "One employee cannot sign in because their SCIM provisioning record is missing. The customer's IdP configuration was changed yesterday. No suspicious activity has been detected.", "identity", criteria=SECURITY, tags=["identity", "seed"]),
        route_case("x_security_same_symptom", "One employee cannot sign in after an attacker obtained their credentials. Security logs show suspicious access from an unknown country and a new API key was created.", "security", criteria=SECURITY, tags=["security", "seed"]),
        route_case("x_negation_outage", "Customer reports what they believe is an outage. Engineering confirmed there is NO platform outage. The problem occurs only in this customer's workspace after a configuration migration.", "product", tags=["negation", "seed"]),
        route_case("x_temporal_fixed", "Login failed yesterday immediately after the customer's SAML certificate expired. The certificate has now been replaced and all users can log in normally. The customer wants help understanding the previous authentication failure.", "identity", tags=["temporal", "seed"]),
        route_case("x_distractor_billing", "The customer is an enterprise account with 450 users. Their annual invoice was paid successfully. The application is functioning normally, but they want a refund for an accidental duplicate charge.", "billing", tags=["billing", "seed"]),
        route_case("x_near_tie", "Customer reports intermittent failures accessing reports. Only their organization has reported the issue so far. Monitoring is inconclusive and no authentication changes have occurred.", "product", tags=["near_tie", "seed"]),
    ]
    specs = [
        ("scope_tenant_api_500", "API requests return HTTP 500 only for one tenant after they changed a workspace setting. Other tenants are healthy and platform error rates are normal.", "product", COMMON, ["scope", "counterfactual"]),
        ("scope_tenant_api_global_500", "API requests return HTTP 500 for hundreds of unrelated tenants across three regions. Platform error rates increased at the same time.", "incident", COMMON, ["scope", "counterfactual"]),
        ("scope_single_reports", "Report exports fail for one enterprise customer. The same reports work for all other customers and monitoring is clean.", "product", COMMON, ["scope", "counterfactual"]),
        ("scope_global_reports", "Report exports fail for customers in North America, Europe, and Asia. More than 90 unrelated organizations opened tickets within 20 minutes.", "incident", COMMON, ["scope", "counterfactual"]),
        ("scope_one_org_latency", "One organization sees 12 second dashboard loads after enabling a custom integration. No other customers report latency.", "product", COMMON, ["scope", "counterfactual"]),
        ("scope_many_org_latency", "Dashboard latency jumped from 200ms to 12 seconds for unrelated organizations in multiple regions at the same timestamp.", "incident", COMMON, ["scope", "counterfactual"]),
        ("near_reports_no_auth_change", "A single organization reports intermittent report failures. Monitoring is inconclusive, no SSO or SCIM changes occurred, and no other customer has reported it.", "product", COMMON, ["near_tie", "product_identity"]),
        ("near_reports_after_idp_change", "A single organization reports intermittent report access failures immediately after its IdP metadata and SCIM group mappings were changed.", "identity", COMMON, ["near_tie", "product_identity"]),
        ("near_exports_role_specific", "Only users with the Finance role cannot export CSV files. Admins and all other roles can export normally.", "product", COMMON, ["near_tie", "product_identity"]),
        ("near_exports_unprovisioned", "Only users newly added through SCIM cannot export CSV files because they were not provisioned into the export-enabled IdP group.", "identity", COMMON, ["near_tie", "product_identity"]),
        ("security_normal_scim", "One employee cannot sign in because their SCIM user record was removed from the IdP group. Security logs show no suspicious activity.", "identity", SECURITY, ["security_identity"]),
        ("security_suspicious_scim", "One employee cannot sign in after suspicious access from an unfamiliar country and a new API key appeared without admin approval.", "security", SECURITY, ["security_identity"]),
        ("security_credential_stuffing", "Several users had successful logins from unfamiliar countries followed by API token creation and changed notification settings.", "security", SECURITY, ["security_identity"]),
        ("security_mfa_rollout", "Users cannot complete login because the customer's MFA policy was rolled out without enrolling the group.", "identity", SECURITY, ["security_identity"]),
        ("billing_duplicate_charge", "The application is healthy, but the customer was charged twice for the same annual renewal and wants a refund.", "billing", COMMON, ["billing_product"]),
        ("billing_feature_bug_paid", "The customer is fully paid and on the correct plan, but the analytics page crashes only in their workspace.", "product", COMMON, ["billing_product"]),
        ("billing_plan_downgrade", "Enterprise analytics disappeared immediately after renewal because the subscription was downgraded to the wrong plan on the invoice.", "billing", COMMON, ["billing_product"]),
        ("billing_permission_not_plan", "Enterprise analytics disappeared for one role because workspace permissions were changed. The subscription and invoice are correct.", "product", COMMON, ["billing_product"]),
        ("trap_old_incident_now_identity", "There was an incident yesterday, now resolved. Today only one customer's SSO users fail because their certificate expired.", "identity", COMMON, ["temporal", "trap"]),
        ("trap_old_identity_now_incident", "A customer had SSO trouble last week, but today unrelated customers in multiple regions cannot authenticate at the same time.", "incident", COMMON, ["temporal", "trap"]),
        ("trap_multi_issue_security_priority", "A customer asks about an invoice and also reports an exposed admin API token with suspicious use. Route the highest-risk issue.", "security", SECURITY, ["priority", "trap"]),
        ("long_buried_identity", "Enterprise account, 900 seats, paid renewal, many integrations, historical billing tickets, custom roles, dashboards, and support escalations. The current issue is that newly hired employees are not provisioned after the IdP metadata changed this morning.", "identity", COMMON, ["long_context"]),
        ("long_buried_product", "Enterprise account, 900 seats, SSO enabled, invoices paid, many admins, historical security review, old incident notes, and several integrations. The current issue is that only the Finance workspace cannot export reports after a role edit.", "product", COMMON, ["long_context"]),
        ("long_buried_incident", "Enterprise account, paid renewal, SSO enabled, old workspace migrations, admin role changes, and historical billing notes. The current issue is simultaneous API timeouts for unrelated customers in several regions.", "incident", COMMON, ["long_context"]),
    ]
    rows.extend(route_case(f"g_{cid}", ticket, expected, criteria=criteria, tags=tags)
                for cid, ticket, expected, criteria, tags in specs)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    rows = cases()
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                         encoding="utf-8")
    print(json.dumps({"path": str(args.path), "cases": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
