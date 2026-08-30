"""
Pruebas del modelo de abandono de clientes.

churn_analysis.py es un guion de arriba abajo, sin funciones: se prueba
ejecutandolo entero sobre una copia del repositorio y midiendo lo que deja
escrito. Es mas lento que probar funciones sueltas, pero es lo unico honesto
aqui, porque lo que este proyecto publica no es codigo -- son siete ficheros
JSON, una tabla en Markdown y un README con cifras impresas.

Lo que se protege, por orden de importancia:

  1. Que las cifras del README siguen siendo las que produce el guion. El
     README anuncia un AUC-ROC de 0,703 como titular del proyecto: si el
     modelo cambia y esa cifra no, el README miente en la portada.
  2. Que la matriz de confusion cuadra con las metricas. Precision y recall
     se derivan de sus cuatro casillas; si no cuadran, una de las dos cosas
     se calculo sobre otra cosa.
  3. Que los cuatro ficheros de segmento comparten forma y no pierden
     clientes por el camino. Los tramos de pd.cut dejan fuera en silencio
     todo lo que cae fuera de los bordes -- sin error, sin aviso: el grafico
     sale con menos clientes de los que hay.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# El README publica estas cifras. La tolerancia existe porque la regresion
# logistica se resuelve numericamente y una version distinta de scikit-learn
# puede mover el ultimo decimal; no esta para tapar un cambio de modelo, que
# moveria las cifras muchisimo mas que esto.
TOLERANCIA = 0.005
PUBLICADO = {
    "auc_roc": 0.703,
    "accuracy": 0.640,
    "recall": 0.615,
    "precision": 0.205,
}

SEGMENTOS = [
    "churn_by_subscription",
    "churn_by_contract",
    "churn_by_tickets",
    "churn_by_charges",
]

CARACTERISTICAS = [
    "Age",
    "MonthlyCharges",
    "TotalUsageHours",
    "SupportTickets",
    "ContractDuration_Months",
    "TenureMonths",
    "NumProducts",
    "SubscriptionType_Premium",
    "SubscriptionType_Standard",
]


@pytest.fixture(scope="module")
def ejecutado(tmp_path_factory):
    """Corre el analisis entero sobre una copia, para no tocar el repositorio."""
    banco = tmp_path_factory.mktemp("churn")
    for fichero in ("churn_analysis.py", "churn_data.csv"):
        shutil.copy(os.path.join(RAIZ, fichero), str(banco))

    entorno = dict(os.environ, MPLBACKEND="Agg")
    proceso = subprocess.run(
        [sys.executable, "churn_analysis.py"],
        cwd=str(banco),
        env=entorno,
        capture_output=True,
        text=True,
    )
    assert proceso.returncode == 0, proceso.stderr[-3000:]
    return banco


def leer(banco, ruta):
    with open(os.path.join(str(banco), ruta), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rendimiento(ejecutado):
    return leer(ejecutado, "data/model_performance.json")


@pytest.fixture(scope="module")
def matriz(ejecutado):
    return leer(ejecutado, "data/confusion_matrix.json")


@pytest.fixture(scope="module")
def importancias(ejecutado):
    return leer(ejecutado, "data/feature_importance.json")


@pytest.fixture(scope="module")
def cortes(ejecutado):
    return {n: leer(ejecutado, f"data/{n}.json") for n in SEGMENTOS}


# --- Lo que promete el README --------------------------------------------


class TestCifrasPublicadas:
    @pytest.mark.parametrize("metrica, valor", sorted(PUBLICADO.items()))
    def test_el_readme_sigue_diciendo_lo_que_produce_el_modelo(
        self, rendimiento, metrica, valor
    ):
        assert rendimiento[metrica] == pytest.approx(valor, abs=TOLERANCIA), (
            f"el README publica {metrica}={valor} y el modelo da "
            f"{rendimiento[metrica]}: hay que actualizar el README"
        )

    def test_la_tasa_de_abandono_publicada_es_la_del_fichero_de_datos(
        self, rendimiento
    ):
        """El README apoya en el 13,1 % todo su argumento sobre por que el
        AUC es la metrica principal y no la accuracy."""
        assert rendimiento["churn_rate"] == pytest.approx(0.131, abs=0.002)

    def test_el_numero_de_clientes_es_el_que_dice_el_informe(self, rendimiento):
        assert rendimiento["n_customers"] == 2000

    def test_el_informe_en_markdown_se_regenera_con_las_cifras_del_modelo(
        self, ejecutado, rendimiento
    ):
        """
        analysis_results.md se escribia a mano y acabo publicando la accuracy
        de un conjunto de datos que ya no existia. Ahora lo genera el guion:
        esta prueba comprueba que de verdad lo hace y que no se quedo pegado.
        """
        texto = open(
            os.path.join(str(ejecutado), "analysis_results.md"), encoding="utf-8"
        ).read()
        assert "Do not edit by hand" in texto
        assert f"{rendimiento['auc_roc']:.4f}" in texto
        assert f"{rendimiento['accuracy']:.4f}" in texto
        for caracteristica in CARACTERISTICAS:
            assert caracteristica in texto

    def test_deja_dibujada_la_matriz_de_confusion(self, ejecutado):
        png = os.path.join(str(ejecutado), "confusion_matrix.png")
        assert os.path.exists(png)
        assert os.path.getsize(png) > 1000


# --- Coherencia interna de las metricas -----------------------------------


class TestMetricas:
    @pytest.mark.parametrize(
        "metrica", ["accuracy", "auc_roc", "precision", "recall", "f1", "churn_rate"]
    )
    def test_toda_probabilidad_vive_entre_cero_y_uno(self, rendimiento, metrica):
        assert 0.0 <= rendimiento[metrica] <= 1.0

    def test_el_modelo_es_mejor_que_tirar_una_moneda(self, rendimiento):
        """Un AUC de 0,5 es azar. Por debajo, el modelo esta al reves."""
        assert rendimiento["auc_roc"] > 0.5

    def test_el_reparto_de_entrenamiento_y_prueba_suma_todos_los_clientes(
        self, rendimiento
    ):
        assert (
            rendimiento["train_size"] + rendimiento["test_size"]
            == rendimiento["n_customers"]
        )

    def test_el_conjunto_de_prueba_es_la_cuarta_parte(self, rendimiento):
        proporcion = rendimiento["test_size"] / rendimiento["n_customers"]
        assert proporcion == pytest.approx(0.25, abs=0.01)

    def test_la_matriz_reparte_exactamente_el_conjunto_de_prueba(
        self, matriz, rendimiento
    ):
        suma = matriz["tn"] + matriz["fp"] + matriz["fn"] + matriz["tp"]
        assert suma == rendimiento["test_size"]

    def test_ninguna_casilla_de_la_matriz_es_negativa(self, matriz):
        assert all(v >= 0 for v in matriz.values())

    def test_el_recall_publicado_sale_de_la_matriz(self, matriz, rendimiento):
        esperado = matriz["tp"] / (matriz["tp"] + matriz["fn"])
        assert rendimiento["recall"] == pytest.approx(esperado, abs=0.001)

    def test_la_precision_publicada_sale_de_la_matriz(self, matriz, rendimiento):
        esperado = matriz["tp"] / (matriz["tp"] + matriz["fp"])
        assert rendimiento["precision"] == pytest.approx(esperado, abs=0.001)

    def test_la_accuracy_publicada_sale_de_la_matriz(self, matriz, rendimiento):
        aciertos = matriz["tp"] + matriz["tn"]
        total = sum(matriz.values())
        assert rendimiento["accuracy"] == pytest.approx(aciertos / total, abs=0.001)

    def test_el_f1_es_la_media_armonica_de_los_otros_dos(self, rendimiento):
        p, r = rendimiento["precision"], rendimiento["recall"]
        assert rendimiento["f1"] == pytest.approx(2 * p * r / (p + r), abs=0.001)

    def test_el_modelo_predice_abandono_alguna_vez(self, matriz):
        """Con class_weight='balanced' no deberia colapsar a la clase mayoritaria,
        pero si lo hiciera, precision y recall saldrian 0 y el panel publicaria
        un modelo que no sirve para nada sin decirlo."""
        assert matriz["tp"] + matriz["fp"] > 0
        assert matriz["tp"] > 0


# --- Importancia de las variables ----------------------------------------


class TestImportancias:
    def test_hay_un_coeficiente_por_variable_del_modelo(self, importancias):
        assert {f["feature"] for f in importancias} == set(CARACTERISTICAS)
        assert len(importancias) == len(CARACTERISTICAS)

    def test_todos_los_coeficientes_son_numeros_finitos(self, importancias):
        for f in importancias:
            assert isinstance(f["coefficient"], (int, float))
            assert abs(f["coefficient"]) < 100

    def test_vienen_ordenados(self, importancias):
        valores = [f["coefficient"] for f in importancias]
        assert valores == sorted(valores)

    def test_un_contrato_mas_largo_retiene(self, importancias):
        """
        Es la unica conclusion de negocio que el README subraya. Si el signo se
        diera la vuelta, el informe seguiria imprimiendose igual de bonito
        recomendando exactamente lo contrario.
        """
        coeficientes = {f["feature"]: f["coefficient"] for f in importancias}
        assert coeficientes["ContractDuration_Months"] < 0

    def test_mas_incidencias_de_soporte_empujan_a_la_puerta(self, importancias):
        coeficientes = {f["feature"]: f["coefficient"] for f in importancias}
        assert coeficientes["SupportTickets"] > 0


# --- Los cuatro cortes por segmento --------------------------------------


class TestSegmentos:
    @pytest.mark.parametrize("nombre", SEGMENTOS)
    def test_los_cuatro_comparten_exactamente_la_misma_forma(self, cortes, nombre):
        """
        Emitir una clave distinta por fichero -- ticket_bucket aqui,
        charges_bucket alla -- es lo que dejo tres de los cuatro graficos sin
        etiquetas en el eje.
        """
        for fila in cortes[nombre]:
            assert set(fila) == {"segment", "total", "churned", "churn_rate"}

    @pytest.mark.parametrize("nombre", SEGMENTOS)
    def test_la_etiqueta_es_texto_imprimible(self, cortes, nombre):
        for fila in cortes[nombre]:
            assert isinstance(fila["segment"], str)
            assert fila["segment"].strip()
            assert fila["segment"] != "nan"

    @pytest.mark.parametrize("nombre", SEGMENTOS)
    def test_la_tasa_es_la_division_de_las_otras_dos_columnas(self, cortes, nombre):
        for fila in cortes[nombre]:
            assert fila["churn_rate"] == pytest.approx(
                fila["churned"] / fila["total"], abs=0.0001
            )

    @pytest.mark.parametrize("nombre", SEGMENTOS)
    def test_ningun_segmento_pierde_clientes_por_los_bordes(
        self, cortes, nombre, rendimiento
    ):
        """
        pd.cut manda a NaN todo lo que cae fuera de los tramos y el groupby lo
        descarta sin decir nada: el grafico sale con menos clientes de los que
        hay, y nadie lo nota mirandolo.
        """
        assert (
            sum(f["total"] for f in cortes[nombre]) == rendimiento["n_customers"]
        ), f"{nombre} no cubre a todos los clientes"

    @pytest.mark.parametrize("nombre", SEGMENTOS)
    def test_nunca_se_van_mas_clientes_de_los_que_hay(self, cortes, nombre):
        for fila in cortes[nombre]:
            assert 0 <= fila["churned"] <= fila["total"]

    @pytest.mark.parametrize("nombre", SEGMENTOS)
    def test_los_abandonos_por_segmento_suman_el_total_de_abandonos(
        self, cortes, nombre, rendimiento
    ):
        esperados = round(rendimiento["churn_rate"] * rendimiento["n_customers"])
        assert sum(f["churned"] for f in cortes[nombre]) == esperados

    @pytest.mark.parametrize(
        "nombre, orden",
        [
            ("churn_by_subscription", ["Basic", "Standard", "Premium"]),
            ("churn_by_contract", ["Monthly", "Annual", "2-Year"]),
            ("churn_by_tickets", ["0", "1-2", "3-4", "5-6", "7+"]),
            (
                "churn_by_charges",
                ["$0-40", "$40-60", "$60-80", "$80-100", "$100-120", "$120+"],
            ),
        ],
    )
    def test_los_tramos_salen_en_el_orden_del_eje_y_no_alfabetico(
        self, cortes, nombre, orden
    ):
        """
        Ordenar alfabeticamente pondria $100-120 antes que $40-60 y el grafico
        se leeria al reves.
        """
        etiquetas = [f["segment"] for f in cortes[nombre]]
        assert etiquetas == [e for e in orden if e in etiquetas]

    def test_el_contrato_mensual_pierde_mas_que_el_de_dos_annos(self, cortes):
        """La conclusion que el README convierte en recomendacion comercial."""
        por_tramo = {f["segment"]: f["churn_rate"] for f in cortes["churn_by_contract"]}
        assert por_tramo["Monthly"] > por_tramo["2-Year"]
