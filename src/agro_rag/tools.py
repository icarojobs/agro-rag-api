from langchain_core.tools import tool


@tool
def liming_calculator(v1: float, v2: float, ctc: float, prnt: float) -> str:
    """Calcula a necessidade de calagem (t/ha) pelo método da saturação por bases.

    Args:
        v1: saturação por bases atual do solo, em %.
        v2: saturação por bases desejada para a cultura, em %.
        ctc: CTC a pH 7 do solo, em cmolc/dm³.
        prnt: PRNT do calcário, em %.
    """
    if not (0 <= v1 <= 100 and 0 <= v2 <= 100):
        return "Erro: V1 e V2 devem estar entre 0 e 100%."
    if ctc <= 0 or not 0 < prnt <= 150:
        return "Erro: CTC deve ser positiva e PRNT deve estar entre 0 e 150%."
    if v1 >= v2:
        return f"V1 ({v1:g}%) já atinge a meta de V2 ({v2:g}%): não há necessidade de calagem."
    dose = (v2 - v1) * ctc / prnt
    return f"NC = ({v2:g} - {v1:g}) x {ctc:g} / {prnt:g} = {dose:.2f} t/ha de calcário."
