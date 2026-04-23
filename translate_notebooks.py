#!/usr/bin/env python3
"""
Systematic PT->EN translation of all Portuguese strings in Jupyter notebooks.
Replaces comments, print statements, docstrings, and markdown cells.
"""
import json
import os
import re

# Comprehensive PT -> EN translation map
TRANSLATIONS = {
    # Common verbs/actions in comments
    "# Carregar o arquivo CSV": "# Load the CSV file",
    "# Carregar o arquivo Excel": "# Load the Excel file (dictionary)",
    "# Carregar o dataset": "# Load the dataset",
    "# Carregar dados": "# Load data",
    "# Carregar dataset": "# Load dataset",
    "# Carregar dados de treinamento": "# Load training data",
    "# Carregar dados de ambos os modelos": "# Load data from both models",
    "# Criar lista para armazenar os matches": "# Create list to store matches",
    "# Criar novo dataframe com as transformações": "# Create new dataframe with transformations",
    "# Cruzar os nomes das colunas com as variáveis do dicionário": "# Cross-reference column names with dictionary variables",
    "# Procurar por matches (case insensitive)": "# Search for matches (case insensitive)",
    "# Salvar os resultados no arquivo": "# Save results to file",
    "# Salvar resultado": "# Save result",
    "# Salvar novo CSV": "# Save new CSV",
    "# Obter todas as colunas do CSV": "# Get all columns from CSV",
    "# Se encontrou match, adicionar à lista": "# If match found, add to list",
    "# Verificar quais variáveis selecionadas existem no dataset": "# Check which selected variables exist in the dataset",
    "# Verificar se existe o diretório de destino": "# Check if destination directory exists",
    "# Filtrar dataset com variáveis disponíveis": "# Filter dataset with available variables",
    "# Calcular estatísticas": "# Calculate statistics",
    "# Calcular missing value percentage for each column": "# Calculate missing value percentage for each column",
    "# Identificar variáveis excluídas": "# Identify excluded variables",
    "# Adicionar todas as outras colunas que não foram processadas": "# Add all other columns that were not processed",
    "# Adicionar as colunas restantes ao dataset final": "# Add remaining columns to the final dataset",
    "# Definir caminhos": "# Define paths",
    "# Preparar dados": "# Prepare data",
    "# Configurar modelo": "# Configure model",
    "# Configurações de cross-validation": "# Cross-validation settings",
    "# Treinar modelo": "# Train model",
    "# Extrair coeficientes": "# Extract coefficients",
    "# Executar": "# Execute",

    # Print statements - common patterns
    "Dataset carregado com sucesso!": "Dataset loaded successfully!",
    "Carregando dataset original...": "Loading original dataset...",
    "Carregando dataset...": "Loading dataset...",
    "Salvando dataset filtrado...": "Saving filtered dataset...",
    "Gerando relatório...": "Generating report...",
    "PROCESSAMENTO CONCLUÍDO COM SUCESSO!": "PROCESSING COMPLETED SUCCESSFULLY!",
    "Dataset original:": "Original dataset:",
    "Dataset filtrado:": "Filtered dataset:",
    "Dataset final:": "Final dataset:",
    "Variáveis removidas:": "Variables removed:",
    "Arquivos gerados:": "Files generated:",
    "variável(es) selecionada(s) não foi(ram) encontrada(s) no dataset original.": "selected variable(s) not found in the original dataset.",
    "Verifique o relatório para detalhes.": "Check the report for details.",
    "Erro: Arquivo não encontrado em": "Error: File not found at",
    "Verifique se o caminho está correto.": "Check if the path is correct.",
    "Erro durante o processamento:": "Error during processing:",
    "Iniciando Feature Engineering": "Starting Feature Engineering",
    "FEATURE ENGINEERING CONCLUÍDO!": "FEATURE ENGINEERING COMPLETED!",
    "Variáveis originais processadas:": "Original variables processed:",
    "Variáveis transformadas criadas:": "Transformed variables created:",
    "Variáveis restantes mantidas:": "Remaining variables kept:",
    "Total de variáveis no dataset final:": "Total variables in final dataset:",
    "Colunas no dataset final:": "Columns in final dataset:",
    "VERIFICAÇÕES FINAIS:": "FINAL CHECKS:",
    "Missing values encontrados:": "Missing values found:",
    "Nenhum missing value no dataset final": "No missing values in final dataset",
    "Tipos de dados:": "Data types:",
    "Adicionadas": "Added",
    "variáveis restantes sem modificação:": "remaining variables without modification:",
    "... e mais": "... and",
    "variáveis": "variables",

    # Variable analysis
    "ANÁLISE DA VARIÁVEL:": "VARIABLE ANALYSIS:",
    "Tipo de dados:": "Data type:",
    "Valores únicos:": "Unique values:",
    "Valores missing:": "Missing values:",
    "Estatísticas descritivas:": "Descriptive statistics:",
    "Distribuição de valores:": "Value distribution:",
    "valores mais frequentes:": "most frequent values:",
    "ANÁLISE EXPLORATÓRIA CONCLUÍDA": "EXPLORATORY ANALYSIS COMPLETED",
    "PRÓXIMOS PASSOS:": "NEXT STEPS:",
    "Revisar as distribuições das variáveis acima": "Review variable distributions above",
    "Confirmar as transformações a serem aplicadas": "Confirm transformations to be applied",
    "Executar o script de feature engineering": "Execute feature engineering script",
    "Salvar resultado em:": "Save result to:",

    # Feature engineering specific
    "mantido como identificador": "kept as identifier",
    "convertido em": "converted to",
    "variáveis dummy": "dummy variables",
    "Regiões:": "Regions:",
    "convertido para": "converted to",
    "mantido como": "kept as",
    "mantido com renomeação": "kept with renaming",
    "criada": "created",
    "Missing values preenchidos com mediana:": "Missing values filled with median:",
    "categorias pequenas agrupadas em": "small categories grouped into",
    "Categorias:": "Categories:",
    "convertido em matriz + variável binária": "converted to matrix + binary variable",
    "Categorias matriz:": "Matrix categories:",
    "'Usou' inclui:": "'Used' includes:",
    "'Não usou' inclui:": "'Not used' includes:",
    "casos": "cases",

    # Transformation descriptions
    "Informações básicas": "Basic information",
    "Se é numérica, mostrar estatísticas descritivas": "If numeric, show descriptive statistics",
    "Se tem poucos valores únicos, mostrar contagem": "If few unique values, show counts",
    "Se é categórica ou tem poucos valores únicos": "If categorical or few unique values",
    "Se tem muitos valores únicos, mostrar apenas os mais frequentes": "If many unique values, show only most frequent",

    # Docstrings
    "Analisa uma variável específica do dataset": "Analyzes a specific variable from the dataset",
    "Feature Engineering das primeiras": "Feature Engineering of the first",
    "Filtra o dataset original mantendo apenas as variáveis selecionadas": "Filters the original dataset keeping only selected variables",
    "e gera relatório das alterações": "and generates a report of changes",

    # Section headers in comments
    "# Lista das primeiras": "# List of first",
    "# Lista de variáveis selecionadas": "# List of selected variables",
    "# Identificação e características demográficas": "# Identification and demographic characteristics",
    "# Dados perinatais e nascimento": "# Perinatal and birth data",
    "# Condições congênitas e síndromes": "# Congenital conditions and syndromes",
    "# Características maternas - educação": "# Maternal characteristics - education",
    "# Pré-natal e gestação": "# Prenatal and pregnancy",
    "# Aleitamento materno e práticas alimentares iniciais": "# Breastfeeding and early feeding practices",
    "# Variáveis socioeconômicas e de equidade": "# Socioeconomic and equity variables",
    "# Antropometria materna atual": "# Current maternal anthropometry",
    "# Indicadores socioeconômicos": "# Socioeconomic indicators",
    "# Variáveis target/outcome": "# Target/outcome variables",

    # Variable descriptions in comments
    "# Código de identificação da criança": "# Child identification code",
    "# Macrorregião": "# Macroregion",
    "# Sexo da criança": "# Child sex",
    "# Idade em anos completos da criança": "# Child age in complete years",
    "# Idade em anos completos da mãe": "# Mother age in complete years",
    "# Cor ou raça da criança": "# Child race/ethnicity",
    "# Com quantas semanas de gravidez a criança nasceu": "# Gestational age at birth (weeks)",
    "# Peso ao nascer (em gramas) da criança": "# Birth weight (grams)",
    "# Com quantos centímetros a criança nasceu": "# Birth length (cm)",
    "# Tipo de parto da criança": "# Delivery type",
    "# Desde o nascimento da criança até hoje": "# From birth until today",

    # Regularization and ML
    "ANÁLISE COMPARATIVA DE REGULARIZAÇÃO": "COMPARATIVE REGULARIZATION ANALYSIS",
    "ABORDAGEM 1: REGULARIZAÇÃO PADRÃO": "APPROACH 1: STANDARD REGULARIZATION",
    "ABORDAGEM 2: FEATURE ENGINEERING BASEADA EM GRUPOS": "APPROACH 2: GROUP-BASED FEATURE ENGINEERING",
    "RESULTADOS COMPARATIVOS": "COMPARATIVE RESULTS",
    "ANÁLISE DE FEATURE SELECTION": "FEATURE SELECTION ANALYSIS",
    "RECOMENDAÇÃO METODOLÓGICA": "METHODOLOGICAL RECOMMENDATION",
    "Ranking por AUC-ROC médio:": "Ranking by mean AUC-ROC:",
    "MÉTODO RECOMENDADO:": "RECOMMENDED METHOD:",
    "Avaliando": "Evaluating",
    "features selecionadas": "features selected",
    "Features após engenharia:": "Features after engineering:",
    "Original:": "Original:",
    "Engenharia:": "Engineered:",
    "Redução:": "Reduction:",
    "Grupos de features identificados:": "Feature groups identified:",
    "Calcula intervalos de confiança usando distribuição t": "Calculates confidence intervals using t-distribution",
    "Avalia um método de regularização com nested cross-validation": "Evaluates a regularization method with nested cross-validation",
    "Implementa análise comparativa de três abordagens de regularização": "Implements comparative analysis of three regularization approaches",

    # ML Algorithm Selection
    "MACHINE LEARNING ALGORITHM SELECTION - MODELOS 1 E 2": "MACHINE LEARNING ALGORITHM SELECTION - MODELS 1 AND 2",
    "Seleção de algoritmos de machine learning para ambos os modelos": "Machine learning algorithm selection for both models",
    "usando nested cross-validation e múltiplos algoritmos": "using nested cross-validation and multiple algorithms",
    "MODELO 1 - Fatores Demográficos/Perinatais:": "MODEL 1 - Demographic/Perinatal Factors:",
    "MODELO 2 - Práticas de Alimentação:": "MODEL 2 - Feeding Practices:",
    "Observações:": "Observations:",
    "Features:": "Features:",
    "Obesos:": "Obese:",
    "AVALIAÇÃO DE ALGORITMOS - MODELO 1": "ALGORITHM EVALUATION - MODEL 1",
    "AVALIAÇÃO DE ALGORITMOS - MODELO 2": "ALGORITHM EVALUATION - MODEL 2",
    "MELHORES ALGORITMOS POR MODELO": "BEST ALGORITHMS PER MODEL",
    "Melhor algoritmo:": "Best algorithm:",
    "ANÁLISE COMPARATIVA GERAL": "GENERAL COMPARATIVE ANALYSIS",
    "Modelo 2 apresenta melhor performance": "Model 2 shows better performance",
    "Modelo 1 apresenta melhor performance": "Model 1 shows better performance",
    "Limitações observadas:": "Observed limitations:",
    "Ambos os modelos apresentam baixa precision": "Both models show low precision",
    "AUC-ROC próximo a 0.6 indica capacidade preditiva limitada": "AUC-ROC near 0.6 indicates limited predictive capacity",
    "Alto número de falsos positivos em ambos os casos": "High number of false positives in both cases",
    "RANKING GERAL DE ALGORITMOS": "OVERALL ALGORITHM RANKING",
    "Posição": "Position",
    "Algoritmo": "Algorithm",
    "Modelo": "Model",
    "Avalia um algoritmo usando nested cross-validation": "Evaluates an algorithm using nested cross-validation",
    "Erro no fold": "Error in fold",

    # Validation
    "VALIDAÇÃO FINAL DOS TRÊS MODELOS DE PREDIÇÃO DE OBESIDADE INFANTIL": "FINAL VALIDATION OF THREE CHILDHOOD OBESITY PREDICTION MODELS",
    "Validação final dos três modelos com hold-out test sets": "Final validation of three models with hold-out test sets",
    "Objetivo: Confirmar que nenhum modelo supera o acaso estatisticamente": "Objective: Confirm that no model statistically exceeds chance",
    "VALIDAÇÃO FINAL -": "FINAL VALIDATION -",
    "Dados de treinamento:": "Training data:",
    "Dados de teste:": "Test data:",
    "Treinando modelo final...": "Training final model...",
    "Preprocessamento: RobustScaler": "Preprocessing: RobustScaler",
    "Balanceamento:": "Balancing:",
    "Resultados no test set (single run):": "Test set results (single run):",
    "Calculando intervalos de confiança com bootstrap": "Calculating bootstrap confidence intervals",
    "RESULTADOS FINAIS COM IC 95% -": "FINAL RESULTS WITH 95% CI -",
    "Métrica": "Metric",
    "Valor": "Value",
    "Interpretação": "Interpretation",
    "Inclui acaso": "Includes chance",
    "Fraco": "Weak",
    "Moderado": "Moderate",
    "ANÁLISE CRÍTICA -": "CRITICAL ANALYSIS -",
    "Capacidade Preditiva:": "Predictive Capacity:",
    "NÃO supera o acaso estatisticamente": "Does NOT statistically exceed chance",
    "Supera o acaso estatisticamente": "Statistically exceeds chance",
    "Limite inferior:": "Lower bound:",
    "Diferença do acaso:": "Difference from chance:",
    "Utilidade Clínica:": "Clinical Utility:",
    "falsos positivos": "false positives",
    "Clinicamente inútil": "Clinically useless",
    "Comparação com Cross-Validation:": "Comparison with Cross-Validation:",
    "CV AUC esperado:": "Expected CV AUC:",
    "Hold-out AUC:": "Hold-out AUC:",
    "Diferença:": "Difference:",
    "RESUMO COMPARATIVO FINAL": "FINAL COMPARATIVE SUMMARY",
    "Supera Acaso?": "Exceeds Chance?",
    "NÃO": "NO",
    "SIM": "YES",
    "CONCLUSÃO CIENTÍFICA GERAL": "GENERAL SCIENTIFIC CONCLUSION",
    "EVIDÊNCIA ROBUSTA DE LIMITAÇÕES PREDITIVAS:": "ROBUST EVIDENCE OF PREDICTIVE LIMITATIONS:",
    "TODOS os modelos falharam em superar o acaso estatisticamente": "ALL models failed to statistically exceed chance",
    "Fatores dos primeiros 24 meses inadequados para predição de obesidade": "First 24 months factors inadequate for obesity prediction",
    "Validação hold-out confirma resultados negativos do cross-validation": "Hold-out validation confirms negative cross-validation results",
    "Precision consistentemente <5% indica alta taxa de falsos positivos": "Precision consistently <5% indicates high false positive rate",
    "Implicações para Pesquisa e Política:": "Implications for Research and Policy:",
    "Paradigma de predição precoce deve ser questionado": "Early prediction paradigm should be questioned",
    "Foco deve migrar para fatores proximais modificáveis": "Focus should shift to modifiable proximal factors",
    "Intervenções baseadas nestes preditores seriam ineficazes": "Interventions based on these predictors would be ineffective",
    "Resultados negativos são cientificamente valiosos": "Negative results are scientifically valuable",

    # Cria/Creates
    "Cria pipeline do modelo baseado no algoritmo especificado": "Creates model pipeline based on specified algorithm",
    "Calcula todas as métricas de performance": "Calculates all performance metrics",
    "Calcula intervalos de confiança 95% usando bootstrap": "Calculates 95% confidence intervals using bootstrap",
    "Valida um modelo específico": "Validates a specific model",
    "Imprime tabela formatada de resultados": "Prints formatted results table",
    "Análise crítica dos resultados": "Critical analysis of results",
    "Executa validação de todos os três modelos": "Executes validation of all three models",
    "Resumo comparativo final dos três modelos": "Final comparative summary of three models",

    # Feature IC analysis
    "Análise completa de feature selection para os três modelos": "Complete feature selection analysis for three models",
    "Analisa um modelo específico extraindo coeficientes significativos": "Analyzes a specific model extracting significant coefficients",
    "Caminhos dos datasets": "Dataset paths",
    "Descrições das variáveis em inglês": "Variable descriptions in English",
    "Corrigir tipos de dados para variáveis binárias": "Fix data types for binary variables",

    # Report generation
    "RELATÓRIO DE SELEÇÃO DE VARIÁVEIS - PRIMEIROS 24 MESES DE VIDA": "VARIABLE SELECTION REPORT - FIRST 24 MONTHS OF LIFE",
    "RESUMO ESTATÍSTICO:": "STATISTICAL SUMMARY:",
    "Total de colunas no dataset original": "Total columns in original dataset",
    "Total de variáveis selecionadas": "Total selected variables",
    "Total de variáveis disponíveis e mantidas": "Total available and retained variables",
    "Total de variáveis excluídas": "Total excluded variables",
    "Dimensões do dataset original": "Original dataset dimensions",
    "Dimensões do dataset filtrado": "Filtered dataset dimensions",
    "VARIÁVEIS SELECIONADAS MAS NÃO ENCONTRADAS NO DATASET:": "SELECTED VARIABLES NOT FOUND IN DATASET:",
    "VARIÁVEIS MANTIDAS NO DATASET FILTRADO:": "VARIABLES KEPT IN FILTERED DATASET:",
    "VARIÁVEIS EXCLUÍDAS DO DATASET ORIGINAL": "VARIABLES EXCLUDED FROM ORIGINAL DATASET",

    # Misc
    "Hiperparâmetros otimizados para": "Optimized hyperparameters for",
    "Amostragem bootstrap": "Bootstrap sampling",
    "Caso especial: só uma classe presente": "Special case: only one class present",
    "Só negativos": "Only negatives",
    "Só positivos": "Only positives",
    "Confusion matrix para specificity e NPV": "Confusion matrix for specificity and NPV",
    "Métricas básicas": "Basic metrics",
    "Predições no test set": "Test set predictions",
    "Algoritmo não reconhecido:": "Algorithm not recognized:",
    "Em caso de erro, usar métricas originais": "In case of error, use original metrics",
    "Consolidar resultados": "Consolidate results",
    "Ordenar por AUC médio": "Sort by mean AUC",

    # AvaliandoColunas specific
    "Análise de regularização para Modelo": "Regularization analysis for Model",
    "Corrigir tipos de dados do Modelo": "Fix data types for Model",
    "Definir grupos de features para análise": "Define feature groups for analysis",
    "Parâmetros para grid search": "Parameters for grid search",
    "Inverso de alpha": "Inverse of alpha",
    "Região: Norte/Nordeste vs Sul/Sudeste/Centro-Oeste": "Region: North/Northeast vs South/Southeast/Central-West",
    "divisão epidemiológica": "epidemiological division",
    "divisão epidemiológica comum": "common epidemiological division",
    "Criar features agregadas para variáveis categóricas": "Create aggregated features for categorical variables",
    "Avaliar métodos na versão engineered": "Evaluate methods on engineered version",
    "Identificar melhor método": "Identify best method",

    # Markdown cells
    "para análise": "for analysis",
    "incluindo": "including",
    "Criando diretório:": "Creating directory:",
}

def translate_notebook(filepath):
    """Translate Portuguese content in a notebook file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Sort translations by length (longest first) to avoid partial replacements
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)

    for pt, en in sorted_translations:
        content = content.replace(pt, en)

    # Also handle emoji prefixes common in the notebooks
    content = content.replace("📊", "[ANALYSIS]")
    content = content.replace("📋", "[INFO]")
    content = content.replace("📁", "[DIR]")
    content = content.replace("✓", "[OK]")
    content = content.replace("⚠️", "[WARNING]")
    content = content.replace("❌", "[FAIL]")
    content = content.replace("🔍", "[SEARCH]")

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


# Target files
target_files = [
    "C-Feature Engeneering/1FeatureEngeneering1-10.ipynb",
    "C-Feature Engeneering/3FeatureEngeneering20-30.ipynb",
    "C-Feature Engeneering/4FeatureEngeering30-40.ipynb",
    "C-Feature Engeneering/5FeatureEngeneering40-50.ipynb",
    "D-Train-Test Split/AvaliandoColunas.ipynb",
    "E-FeatureSelection/FSMODEL1.ipynb",
    "E-FeatureSelection/FSMODEL2.ipynb",
    "E-FeatureSelection/FSMODEL3.ipynb",
    "E-FeatureSelection/modelando1e2.ipynb",
    "E-FeatureSelection/modelando3.ipynb",
    "F-Modelo vencedor com holdOut/validacao.ipynb",
]

base = "/Users/marcelosilva/project/early-obesity-prediction"

for f in target_files:
    path = os.path.join(base, f)
    if os.path.exists(path):
        changed = translate_notebook(path)
        print(f"{'TRANSLATED' if changed else 'NO CHANGES':>12}  {f}")
    else:
        print(f"{'NOT FOUND':>12}  {f}")

# SECOND PASS - more translations
TRANSLATIONS2 = {
    # Comment patterns
    "# Remover a coluna": "# Remove the column",
    "# Remover apenas valores impossíveis": "# Remove only impossible values",
    "# Verificar distribuição": "# Check distribution",
    "# Verificar se": "# Check if",
    "# Comparar proporções": "# Compare proportions",
    "# Comparação com": "# Comparison with",
    "# Converter matriz binária": "# Convert binary matrix",
    "# Corrigir tipos de dados": "# Fix data types",
    "# Criar variável target": "# Create target variable",
    "# Criar variables binárias": "# Create binary variables",
    "# Criar o novo dataset": "# Create the new dataset",
    "# Colunas COM missing": "# Columns WITH missing",
    "# Colunas SEM missing": "# Columns WITHOUT missing",
    "# Calcular dados faltantes": "# Calculate missing data",
    "# Calcular obesidade": "# Calculate obesity",
    "# Analisar distribuição": "# Analyze distribution",
    "# Analisar missing values": "# Analyze missing values",
    "# Adicionar variables target": "# Add target variables",
    "# Análise por tipo de dados": "# Analysis by data type",
    "# Consolidar: média se ambos": "# Consolidate: mean if both",
    "# Exibir algumas linhas": "# Display some rows",
    "# Exibir informações básicas": "# Display basic information",
    "# Informações sobre o novo dataset": "# Information about the new dataset",
    "# Inserir novas variables": "# Insert new variables",
    "# Load the dataset transformado": "# Load the transformed dataset",
    "# Se a coluna não deve ser removida": "# If column should not be removed",
    "# Como não há valores nulos": "# Since there are no null values",
    "# Mostrar até 10 valores mais frequentes": "# Show up to 10 most frequent values",
    "# Para categóricas, mostrar distribuição": "# For categorical, show distribution",
    "# Para variables com poucos valores únicos": "# For variables with few unique values",
    "# Subset sem missing em NENHUMA das colunas com missing": "# Subset without missing in ANY of the missing columns",
    "# CRIAÇÃO DO DATASET MODELO 1": "# MODEL 1 DATASET CREATION",
    "# CRIAÇÃO DO DATASET MODELO 2": "# MODEL 2 DATASET CREATION",
    "# CRIAÇÃO DO DATASET MODELO 3": "# MODEL 3 DATASET CREATION",
    "# Mantido para identificação posterior": "# Kept for later identification",

    # Variable descriptions in tuples
    "Binária: busca informações": "Binary: seeks information",
    "Binária: recebe benefício": "Binary: receives benefit",
    "Binária: uso de mamadeira": "Binary: bottle feeding use",
    "Binária:": "Binary:",
    "Escore econômico (preservado)": "Economic score (preserved)",
    "Z-score IMC (target - apenas imputação)": "BMI z-score (target - imputation only)",
    "Transforma a variável vd_zimc": "Transforms the variable vd_zimc",

    # Print strings
    "Média:": "Mean:",
    "Mediana:": "Median:",
    "Média de ambas:": "Mean of both:",
    "para binárias": "for binary",

    # AvaliandoColunas specific
    "colunas_com_missing": "cols_with_missing",
    "colunas_sem_missing": "cols_without_missing",
    "colunas_missing_sorted": "missing_cols_sorted",
    "modelo2_sample_size": "model2_sample_size",
    "modelo2_percent": "model2_percent",
    "percent_retido": "percent_retained",
    "count_modelo2": "count_model2",
    "df_modelo1": "df_model1",
    "df_modelo2": "df_model2",
    "df_complete_cases": "df_complete_cases",

    # More comments
    "# Análise de Regularização": "# Regularization Analysis",
    "# Métricas para avaliação": "# Metrics for evaluation",
    "# Nested cross-validation": "# Nested cross-validation",
    "# Variáveis binárias que precisam de correção": "# Binary variables that need fixing",
    "# Configurações dos modelos baseadas nos melhores resultados": "# Model configurations based on best results",
    "# Hiperparâmetros otimizados para": "# Optimized hyperparameters for",
    "# Pipeline com preprocessamento": "# Pipeline with preprocessing",

    # More prints
    "Executar análise comparativa": "Execute comparative analysis",
    "Executar seleção de algoritmos": "Execute algorithm selection",
    "Executar validação completa": "Execute complete validation",
    "Executar análise completa": "Execute complete analysis",
    "Executar feature engineering": "Execute feature engineering",
    "Executar o processamento": "Execute processing",
}

for f in target_files:
    path = os.path.join(base, f)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            content = fh.read()
        original = content
        sorted_t2 = sorted(TRANSLATIONS2.items(), key=lambda x: len(x[0]), reverse=True)
        for pt, en in sorted_t2:
            content = content.replace(pt, en)
        if content != original:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            print(f"PASS 2 OK  {f}")
        else:
            print(f"PASS 2 --  {f}")
