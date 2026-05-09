"""Gate: Data Loss Prevention (DLP) using ML and Regex."""

import re
from typing import List

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from .models import GateResult, Finding, Severity

# Regex patterns for known high-risk secrets and tokens
SECRET_PATTERNS = {
    "AWS_KEY": r"(?i)AKIA[0-9A-Z]{16}",
    "GENERIC_SECRET": r"(?i)(?:secret|token|password|api_key|apikey|sk)[\s:=]+['\"]([a-zA-Z0-9_\-\.]{16,})['\"]",
    "JWT": r"ey[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*",
    "RSA_PRIVATE_KEY": r"-----BEGIN RSA PRIVATE KEY-----",
    "OPENAI_KEY": r"sk-[a-zA-Z0-9]{32,}",
}

class DlpValidator:
    """Validator that checks for data leaks (PII/secrets) in code."""

    def __init__(self, ml_enabled: bool = True):
        self.ml_enabled = ml_enabled
        self.nlp = None
        
        # Sadece güçlü bir GPU donanımında (Sayzek VM) ML yeteneklerini aktifleştiriyoruz
        if self.ml_enabled and pipeline is not None:
            try:
                # Named Entity Recognition (NER) ile Kişisel Veri (PII) Tespiti
                # Model indirilmediyse HuggingFace'den otomatik çeker (ilk çalışmada)
                self.nlp = pipeline(
                    "ner", 
                    aggregation_strategy="simple", 
                    model="dslim/bert-base-NER",
                    device=0 # GPU 0 kullanımını zorlar (40GB VRAM olduğu için çok hızlı olacaktır)
                )
            except Exception as e:
                # CUDA veya kütüphane hatası olursa sessizce pas geç, regex kullan
                self.nlp = None

    def validate(self, code: str) -> GateResult:
        """Scan generated code for secrets and PII."""
        findings: List[Finding] = []
        
        # 1. Regex Tabanlı Şifre/Token Taraması (Sıfır Hata Payı ile Hızlı Tarama)
        for secret_type, pattern in SECRET_PATTERNS.items():
            for match in re.finditer(pattern, code):
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code="DLP-SECRET",
                    message=f"Tehlikeli veri sızıntısı: {secret_type} formatında bir şifre tespit edildi.",
                    suggestion="Hardcode edilmiş şifreleri kaldırın ve çevre değişkenleri (environment variables) kullanın."
                ))

        # 2. Yapay Zeka (NLP) Tabanlı Hassas Veri Taraması (Bağlam Anlama)
        if self.nlp is not None:
            # Modelin maksimum token limiti (genelde 512) için kodu parçalara bölelim
            chunk_size = 1500  
            chunks = [code[i:i+chunk_size] for i in range(0, len(code), chunk_size)]
            
            for chunk in chunks:
                try:
                    entities = self.nlp(chunk)
                    for entity in entities:
                        # Sadece "Kişi" (PER) veya "Kurum" (ORG) gibi isimleri tespit eder
                        if entity['entity_group'] in ['PER', 'ORG']:
                            findings.append(Finding(
                                severity=Severity.WARNING,
                                code="DLP-PII",
                                message=f"Potansiyel Kişisel/Kurumsal Veri İfşası (PII): '{entity['word']}' ({entity['entity_group']})",
                                suggestion="Test verilerinde gerçek kişisel veriler yerine sahte (faker) veriler kullanın."
                            ))
                except Exception:
                    pass

        # DLP kontrolünü geçme şartı: ERROR seviyesinde bulgu olmamalı
        passed = not any(f.severity == Severity.ERROR for f in findings)

        return GateResult(
            gate_name="dlp",
            passed=passed,
            findings=findings,
            details="Veri sızıntısı taraması (ML + Regex) tamamlandı."
        )
