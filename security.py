"""
Módulo de segurança para validação e sanitização de dados.
"""
import re
import html
from typing import Optional
from fastapi import HTTPException
from config import settings


class SecurityValidator:
    """Classe para validações de segurança."""
    
    # Padrões de texto suspeito
    SUSPICIOUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Scripts maliciosos
        r'javascript:',  # JavaScript inline
        r'on\w+\s*=',  # Event handlers
        r'data:text/html',  # Data URLs
    ]
    
    @staticmethod
    def validate_text_input(text: str) -> str:
        """
        Valida e sanitiza texto de entrada.
        
        Args:
            text: Texto a ser validado
            
        Returns:
            Texto sanitizado
            
        Raises:
            HTTPException: Se o texto for inválido
        """
        if not text or not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="O campo de texto não pode ser vazio."
            )
        
        # Verificar tamanho máximo
        if len(text) > settings.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Texto muito longo. Máximo permitido: {settings.max_text_length} caracteres."
            )
        
        # Sanitizar HTML
        sanitized_text = html.escape(text)
        
        # Verificar padrões suspeitos
        for pattern in SecurityValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="Texto contém conteúdo suspeito ou malicioso."
                )
        
        # Verificar caracteres não imprimíveis excessivos
        non_printable_count = sum(1 for char in text if ord(char) < 32 and char not in '\n\r\t')
        if non_printable_count > len(text) * 0.1:  # Mais de 10% de caracteres não imprimíveis
            raise HTTPException(
                status_code=400,
                detail="Texto contém muitos caracteres não imprimíveis."
            )
        
        return text.strip()
    
    @staticmethod
    def validate_summarization_params(max_length: int, min_length: int) -> None:
        """
        Valida parâmetros de sumarização.
        
        Args:
            max_length: Comprimento máximo
            min_length: Comprimento mínimo
            
        Raises:
            HTTPException: Se os parâmetros forem inválidos
        """
        if max_length <= min_length:
            raise HTTPException(
                status_code=400,
                detail="max_length deve ser maior que min_length"
            )
        
        if max_length > 1000:
            raise HTTPException(
                status_code=400,
                detail="max_length não pode exceder 1000 caracteres"
            )
        
        if min_length < 10:
            raise HTTPException(
                status_code=400,
                detail="min_length deve ser pelo menos 10 caracteres"
            )
    
    @staticmethod
    def validate_method(method: str) -> str:
        """
        Valida método de sumarização.
        
        Args:
            method: Método a ser validado
            
        Returns:
            Método validado
            
        Raises:
            HTTPException: Se o método for inválido
        """
        valid_methods = ["extractive", "abstractive", "auto"]
        if method not in valid_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Método inválido. Escolha entre: {', '.join(valid_methods)}"
            )
        return method


# Instância global do validador
security_validator = SecurityValidator()
