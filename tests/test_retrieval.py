from __future__ import annotations

import unittest

from langchain_core.documents import Document

from financial_rag import FinancialRAG, RAGConfig
from financial_rag.retrieval import (
    expand_query,
    focused_evidence_queries,
    is_ai_risk_synthesis,
    is_exact_financial,
    target_companies,
)


def table_doc(company: str, page: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "company": company,
            "page_number": page,
            "source_file": "test.pdf",
            "doc_type": "table_page",
        },
    )


class TableSupplementTests(unittest.TestCase):
    def engine_with(self, documents: list[Document]) -> FinancialRAG:
        engine = FinancialRAG.__new__(FinancialRAG)
        engine.config = RAGConfig(table_supplement_k=1)
        engine.table_pages = documents
        return engine

    def test_current_ratio_prefers_complete_balance_sheet(self) -> None:
        audit_report = table_doc(
            "Amazon",
            58,
            "Report of Independent Registered Public Accounting Firm. "
            "We audited the consolidated balance sheets.",
        )
        balance_sheet = table_doc(
            "Amazon",
            63,
            "Consolidated Balance Sheets. Total current assets 229083. "
            "Total current liabilities 218005.",
        )
        engine = self.engine_with([audit_report, balance_sheet])
        query = expand_query("What was Amazon's current ratio at December 31, 2025?")

        selected = engine._table_supplements(query, ["Amazon"])

        self.assertEqual(selected[0].metadata["page_number"], 63)

    def test_segment_margin_prefers_page_with_both_inputs(self) -> None:
        revenue_only = table_doc(
            "Amazon",
            43,
            "AWS net sales revenue 128725. International net sales 161894.",
        )
        full_segment_table = table_doc(
            "Amazon",
            109,
            "AWS net sales 128725. AWS operating expenses 83119. "
            "AWS operating income 45606.",
        )
        engine = self.engine_with([revenue_only, full_segment_table])
        query = expand_query("Compare the 2025 operating margins of AWS and Google Cloud.")

        selected = engine._table_supplements(query, ["Amazon"])

        self.assertEqual(selected[0].metadata["page_number"], 109)

    def test_exact_number_prefers_table_containing_that_value(self) -> None:
        unrelated = table_doc(
            "Amazon",
            107,
            "Segment information. North America net sales 426305. International net sales 161894.",
        )
        complete = table_doc(
            "Amazon",
            109,
            "AWS net sales 128725. AWS operating income 45606. Consolidated net sales 716924.",
        )
        engine = self.engine_with([unrelated, complete])
        query = expand_query(
            "Which FY2025 segment reported $45,606M of op inc, what was its revenue, and what margin does that imply?"
        )

        selected = engine._table_supplements(query, ["Amazon"])

        self.assertEqual(selected[0].metadata["page_number"], 109)

    def test_net_income_prefers_consolidated_results_over_pro_forma(self) -> None:
        pro_forma = table_doc(
            "Microsoft",
            118,
            "Unaudited pro forma results. Revenue 247442. Net income 88308.",
        )
        consolidated = table_doc(
            "Microsoft",
            67,
            "Summary results of operations. Revenue 281724. Net income 101832.",
        )
        engine = self.engine_with([pro_forma, consolidated])
        query = expand_query(
            "What is $28.9 billion as a percentage of Microsoft's FY2025 net income?"
        )

        selected = engine._table_supplements(query, ["Microsoft"])

        self.assertEqual(selected[0].metadata["page_number"], 67)

    def test_azure_growth_prefers_explicit_growth_disclosure(self) -> None:
        segment_total = table_doc(
            "Microsoft",
            68,
            "Intelligent Cloud revenue 106265 87464 21 percent.",
        )
        azure_growth = table_doc(
            "Microsoft",
            70,
            "Azure and other cloud services revenue grew 34 percent driven by demand.",
        )
        engine = self.engine_with([segment_total, azure_growth])
        query = expand_query(
            "Which grew faster in FY2025: Azure and other cloud services or Intelligent Cloud?"
        )

        selected = engine._table_supplements(query, ["Microsoft"])

        self.assertEqual(selected[0].metadata["page_number"], 70)

    def test_subsequent_event_prefers_january_unrealized_gain(self) -> None:
        annual_gain = table_doc(
            "Alphabet/Google",
            37,
            "Gain on equity securities net 24080 for year ended December 31 2025.",
        )
        subsequent_event = table_doc(
            "Alphabet/Google",
            89,
            "Subsequent event. In January 2026 we recognized approximately 32000 of unrealized gains in non-marketable investments following observable transactions.",
        )
        engine = self.engine_with([annual_gain, subsequent_event])
        query = expand_query(
            "Can Alphabet's $24.08 billion gain be added to January 2026 unrealized gains?"
        )

        selected = engine._table_supplements(query, ["Alphabet/Google"])

        self.assertEqual(selected[0].metadata["page_number"], 89)

    def test_total_revenue_prefers_complete_company_total(self) -> None:
        cloud_metric = table_doc(
            "Microsoft",
            9,
            "Microsoft Cloud revenue was 168900 for fiscal year 2025.",
        )
        company_total = table_doc(
            "Microsoft",
            143,
            "Segment information. Productivity revenue 120810. Intelligent Cloud revenue 106265. Total Revenue 281724.",
        )
        engine = self.engine_with([cloud_metric, company_total])
        query = expand_query("微软最近财年的总收入是多少？")

        selected = engine._table_supplements(query, ["Microsoft"])

        self.assertEqual(selected[0].metadata["page_number"], 143)


class QueryRoutingTests(unittest.TestCase):
    def test_multilingual_company_aliases(self) -> None:
        self.assertEqual(target_companies("亚马逊的现金是多少？"), ["Amazon"])
        self.assertEqual(target_companies("微软的云收入是多少？"), ["Microsoft"])
        self.assertEqual(target_companies("谷歌的员工人数是多少？"), ["Alphabet/Google"])

    def test_financial_intent_variants(self) -> None:
        self.assertTrue(is_exact_financial("Which segment grew faster?"))
        self.assertTrue(is_exact_financial("Give the signed values and quantify the change."))
        self.assertTrue(is_exact_financial("亚马逊的收入增加了多少？"))

    def test_focused_metric_queries(self) -> None:
        azure = focused_evidence_queries("Compare Azure growth with Intelligent Cloud.")
        subsequent = focused_evidence_queries(
            "Can January 2026 unrealized gains be added to the 2025 amount?"
        )

        self.assertTrue(any("Azure and other cloud services" in query for query in azure))
        self.assertTrue(any("subsequent event" in query for query in subsequent))

    def test_chinese_ai_risk_triggers_focused_routing(self) -> None:
        self.assertTrue(is_ai_risk_synthesis("请比较三家公司披露的主要AI竞争和法律风险。"))


class ResponseMetadataTests(unittest.TestCase):
    def test_stop_reason(self) -> None:
        message = type("Message", (), {"response_metadata": {"stop_reason": "max_tokens"}})()
        self.assertEqual(FinancialRAG._stop_reason(message), "max_tokens")


class PriorityEvidenceTests(unittest.TestCase):
    def engine_with(self, documents: list[Document]) -> FinancialRAG:
        engine = FinancialRAG.__new__(FinancialRAG)
        engine.evidence_pages = documents
        return engine

    def test_priority_page_detection_uses_metric_combinations(self) -> None:
        self.assertTrue(FinancialRAG._is_priority_evidence_page(
            "Federal statutory rate 21%. Items accounting for differences. Total 19.6%.",
            "Amazon",
        ))
        self.assertTrue(FinancialRAG._is_priority_evidence_page(
            "Consolidated statements of cash flows. Investing activities. "
            "Purchases of property and equipment 131819.",
            "Amazon",
        ))
        self.assertFalse(FinancialRAG._is_priority_evidence_page(
            "We expect to invest in property and equipment next year.",
            "Amazon",
        ))

    def test_tax_rate_selects_complete_page_for_each_company(self) -> None:
        docs = [
            table_doc("Alphabet/Google", 85, "Federal statutory rate 21.0%. Effective income tax rate 16.8%."),
            table_doc("Amazon", 105, "Federal statutory rate 21.0%. Items accounting for differences. Total 19.6%."),
            table_doc("Microsoft", 125, "Federal statutory rate 21.0%. Effective rate 17.6%."),
        ]
        engine = self.engine_with(docs)
        selected = engine._evidence_supplements(
            "Compare the effective tax rate with the statutory tax rate.",
            ["Alphabet/Google", "Amazon", "Microsoft"],
        )
        self.assertEqual([doc.metadata["page_number"] for doc in selected], [85, 105, 125])

    def test_cash_tax_selects_paid_and_provision_pages(self) -> None:
        docs = [
            table_doc("Microsoft", 125, "Provision for income taxes 21795. Federal statutory rate. Effective rate."),
            table_doc("Microsoft", 127, "Income taxes paid, net of refunds, were 28.7 billion."),
        ]
        engine = self.engine_with(docs)
        selected = engine._evidence_supplements(
            "Calculate the cash-tax gap: taxes paid minus provision for income taxes.",
            ["Microsoft"],
        )
        self.assertEqual(
            {doc.metadata["page_number"] for doc in selected},
            {125, 127},
        )

    def test_capex_prefers_cash_flow_statement_over_narrative(self) -> None:
        narrative = table_doc(
            "Amazon", 41,
            "Cash capital expenditures were 128.3 billion for infrastructure.",
        )
        cash_flow = table_doc(
            "Amazon", 60,
            "Consolidated statements of cash flows. Investing activities. "
            "Purchases of property and equipment 131819.",
        )
        engine = self.engine_with([narrative, cash_flow])
        selected = engine._evidence_supplements(
            "What were capital expenditures reported as purchases of property and equipment?",
            ["Amazon"],
        )
        self.assertEqual(selected[0].metadata["page_number"], 60)

    def test_capex_keeps_cash_flow_and_revenue_pages(self) -> None:
        docs = [
            table_doc(
                "Microsoft", 91,
                "Consolidated statements of cash flows. Investing activities. "
                "Additions to property and equipment 64551.",
            ),
            table_doc(
                "Microsoft", 92,
                "Property and equipment additions included finance leases and "
                "other noncash investing activities 70000.",
            ),
            table_doc(
                "Microsoft", 63,
                "Our reportable segments are Productivity and Business Processes, "
                "Intelligent Cloud, and More Personal Computing. International "
                "operations provide a significant portion of total revenue.",
            ),
            table_doc(
                "Microsoft", 68,
                "Segment results of operations. Productivity and Business Processes 120810. "
                "Intelligent Cloud 106265. More Personal Computing 54649. "
                "Total revenue $ 281724.",
            ),
        ]
        engine = self.engine_with(docs)
        selected = engine._evidence_supplements(
            "Compute capital expenditures as a percentage of revenue.",
            ["Microsoft"],
        )
        self.assertEqual(
            {doc.metadata["page_number"] for doc in selected},
            {68, 91},
        )


if __name__ == "__main__":
    unittest.main()
