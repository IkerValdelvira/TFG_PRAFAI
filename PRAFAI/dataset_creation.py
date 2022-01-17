import argparse
import math
import os

import pandas as pd
import numpy as np
import dateutil.relativedelta

input_dir = ""
output_dir = ""


def leer_ficheros():
    outcomes = pd.read_excel(os.path.join(input_dir,"Outcomes.xlsx"))
    outcomes.drop(['Columna3'], axis=1, inplace=True)
    outcomes.drop(outcomes.columns[10:], axis=1, inplace=True)

    laboratorio = pd.read_excel(os.path.join(input_dir,"Laboratorio.xlsx"))

    laboratorio_extra_1 = pd.read_excel(os.path.join(input_dir,"LaboratorioGrupo1.xlsx"))
    laboratorio_extra_2 = pd.read_excel(os.path.join(input_dir,"LaboratorioGrupo2.xlsx"))
    laboratorio_extra = pd.concat([laboratorio_extra_1, laboratorio_extra_2])

    pruebas = pd.read_excel(os.path.join(input_dir,"Pruebas.xlsx"))

    procedimientos_quirurgicos = pd.read_excel(os.path.join(input_dir,"ProcedimientosQuirurgicos.xlsx"))

    numero_ingresos = pd.read_excel(os.path.join(input_dir,"NumeroIngresos.xlsx"))

    ultimo_ingreso = pd.read_excel(os.path.join(input_dir,"UltimoIngreso.xlsx"))

    diagnosticos = pd.read_excel(os.path.join(input_dir,"Diagnosticos.xlsx"))

    datos_de_paciente = pd.read_excel(os.path.join(input_dir,"DatosDePaciente.xlsx"), header=1)

    hft_1_1 = pd.read_excel(os.path.join(input_dir,"HFT_1_1.xlsx"))
    hft_1_2 = pd.read_excel(os.path.join(input_dir,"HFT_1_2.xlsx"))
    hft_1_3 = pd.read_excel(os.path.join(input_dir,"HFT_1_3.xlsx"))
    hft_2_1 = pd.read_excel(os.path.join(input_dir,"HFT_2_1.xlsx"))
    hft_2_2 = pd.read_excel(os.path.join(input_dir,"HFT_2_2.xlsx"))
    hft_2_3 = pd.read_excel(os.path.join(input_dir,"HFT_2_3.xlsx"))
    hft = pd.concat([hft_1_1, hft_1_2, hft_1_3, hft_2_1, hft_2_2, hft_2_3])

    talla_peso_imc = pd.read_excel(os.path.join(input_dir,"TallaPesoIMC.xlsx"))

    epoc_1 = pd.read_excel(os.path.join(input_dir,"EPOCgrupo1.xlsx"))
    epoc_2 = pd.read_excel(os.path.join(input_dir,"EPOCgrupo2.xlsx"))
    epoc = pd.concat([epoc_1, epoc_2])

    ficheros = {
        "outcomes": outcomes,
        "laboratorio": laboratorio,
        "laboratorio_extra": laboratorio_extra,
        "pruebas": pruebas,
        "procedimientos_quirurgicos": procedimientos_quirurgicos,
        "numero_ingresos": numero_ingresos,
        "ultimo_ingreso": ultimo_ingreso,
        "diagnosticos": diagnosticos,
        "datos_de_paciente": datos_de_paciente,
        "hft": hft,
        "talla_peso_imc": talla_peso_imc,
        "epoc": epoc
    }

    return ficheros


def cargar_datos_FA(primeros_outcome):
    # Columnas a crear
    id_array = []
    fecha_dx_array = []
    dx_array = []

    for index, row in primeros_outcome.iterrows():
        # Obtenemos datos necesarios por cada fila
        id = row['ID']
        fecha_dx = row['FECHA DX']      # Ya lo recoge en formato Datetime
        dx = row['DX']

        # Establecer tipos de FA:
        #   Fibrilación auricular / Fibrilación auricular no especificada --> 0
        #   Fibrilación paroxística auricular --> 1
        #   Fibrilación persistente auricular --> 2
        #   Fibrilación crónica auricular (permanente) --> 3
        if ((dx == 'FIBRILACION AURICULAR') | (dx == 'FIBRILACIÓN AURICULAR NO ESPECIFICADA') | (dx == 'FI+A294BRILACION AURICULAR') | (dx == 'FI+A653BRILACION AURICULAR')):
            tipo_fa = 0
        elif (dx == 'FIBRILACIÓN PAROXÍSTICA AURICULAR'):
            tipo_fa = 1
        elif (dx == 'FIBRILACIÓN PERSISTENTE AURICULAR'):
            tipo_fa = 2
        elif ((dx == 'FIBRILACIÓN CRÓNICA AURICULAR') | (dx == 'FI+A295BRILACIÓN CRÓNICA AURICULAR')):
            tipo_fa = 3

        # Añadir datos a los arrays
        id_array.append(id)
        fecha_dx_array.append(fecha_dx)
        dx_array.append(tipo_fa)

        # Cargar DataFrame
        data = {'id': id_array,
                'fecha': fecha_dx_array,
                'tipo_fa': dx_array}
        tabla_prafai = pd.DataFrame(data)

    return tabla_prafai


def cargar_laboratorio(laboratorio, tabla_prafai):
    # Columnas a crear (urea, creatinina...)
    urea_array = []
    creatinina_array = []
    albumina_array = []
    glucosa_array = []
    hba1c_array = []
    potasio_array = []
    calcio_array = []
    hdl_array = []
    no_hdl_array = []
    ldl_array = []
    colesterol_array = []
    ntprobnp_array = []
    bnp_array = []
    troponina_tni_array = []
    troponina_tnt_array = []
    acido_urico_array = []
    dimero_d_array = []
    fibrinogeno_array = []
    aldosterona_array = []
    pcr_array = []
    leucocitos_array = []
    pct_array = []
    tsh_array = []
    t4l_array = []

    for index, row in tabla_prafai.iterrows():      # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']

        pruebas = {}            # Se almacenan TODAS las pruebas realizadas a ese paciente ANTES Y DESPUES DEL DIAGNOSTICO DE FA
        medias_pruebas = {}     # Se almacenan LAS MEDIAS de todas pruebas realizadas a ese paciente ANTES Y DESPUES DEL DIAGNOSTICO DE FA

        pruebas_paciente = laboratorio[laboratorio['Id Paciente'] == id]        # Se obtienen todas las pruebas realizadas a ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_prueba, row_prueba in pruebas_paciente.iterrows():        # Se va a recorrer cada prueba del paciente
            id_prueba = row_prueba['Id prueba']     # Se obtiene el id de esa prueba
            resultado_prueba = row_prueba['Resultado texto']    # Se obtiene el resultado de esa prueba
            try:    # Se intenta convertir el resultado a FLOAT, si no es posible no se tiene en cuenta
                resultado_prueba_float = float(resultado_prueba)
                if(resultado_prueba_float < 10000):     # EVITAR VALORES INCORRECTOS (LIMITE 10000)
                    if (id_prueba in pruebas.keys()):
                        # La prueba ya se le ha realizado mas veces al paciente
                        pruebas[id_prueba].append(resultado_prueba_float)  # Se añade el resultado de la prueba
                    else:
                        # Es la primera vez que se le realiza la prueba al paciente
                        pruebas[id_prueba] = [resultado_prueba_float]  # Se guarda su resultado
            except ValueError:
                pass
                #print("PRUEBA NO FLOAT: " + str(resultado_prueba))

        for prueba in pruebas.keys():    # Se recorren todas la pruebas del paciente ANTES DEL DIAGNOSTICO DE FA
            medias_pruebas[prueba] = round(np.mean(pruebas[prueba]),3)   # Se guarda la media de resultados de la prueba realizada al paciente

        # Añadir media de pruebas de UREA al paciente
        if(173 in medias_pruebas.keys()):
            urea_array.append(medias_pruebas[173])
        else:
            urea_array.append(-1)

        # Añadir media de pruebas de CREATININA al paciente
        if (174 in medias_pruebas.keys()):
            creatinina_array.append(medias_pruebas[174])
        else:
            creatinina_array.append(-1)

        # Añadir media de pruebas de ALBUMINA al paciente
        if (309 in medias_pruebas.keys()):
            albumina_array.append(medias_pruebas[309])
        elif (310 in medias_pruebas.keys()):
            albumina_array.append(medias_pruebas[310])
        else:
            albumina_array.append(-1)

        # Añadir media de pruebas de GLUCOSA al paciente
        if (166 in medias_pruebas.keys()):
            glucosa_array.append(medias_pruebas[166])
        else:
            glucosa_array.append(-1)

        # Añadir media de pruebas de HbA1c al paciente
        if (2558 in medias_pruebas.keys()):
            hba1c_array.append(medias_pruebas[2558])
        else:
            hba1c_array.append(-1)

        # Añadir media de pruebas de POTASIO al paciente
        if (192 in medias_pruebas.keys()):
            potasio_array.append(medias_pruebas[192])
        else:
            potasio_array.append(-1)

        # Añadir media de pruebas de CALCIO al paciente
        if (178 in medias_pruebas.keys()):
            calcio_array.append(medias_pruebas[178])
        else:
            calcio_array.append(-1)

        # Añadir media de pruebas de HDL al paciente
        if (170 in medias_pruebas.keys()):
            hdl_array.append(medias_pruebas[170])
        else:
            hdl_array.append(-1)

        # Añadir media de pruebas de NO HDL al paciente
        if (172 in medias_pruebas.keys()):
            no_hdl_array.append(medias_pruebas[172])
        else:
            no_hdl_array.append(-1)

        # Añadir media de pruebas de LDL al paciente
        if (171 in medias_pruebas.keys()):
            ldl_array.append(medias_pruebas[171])
        else:
            ldl_array.append(-1)

        # Añadir media de pruebas de COLESTEROL al paciente
        if (168 in medias_pruebas.keys()):
            colesterol_array.append(medias_pruebas[168])
        else:
            colesterol_array.append(-1)

        # Añadir media de pruebas de NT-proBNP al paciente
        if (6014 in medias_pruebas.keys()):
            ntprobnp_array.append(medias_pruebas[6014])
        else:
            ntprobnp_array.append(-1)

        # Añadir media de pruebas de BNP al paciente
        if (6015 in medias_pruebas.keys()):
            bnp_array.append(medias_pruebas[6015])
        else:
            bnp_array.append(-1)

        # Añadir media de pruebas de TROPONINA (TNI) al paciente
        if (6012 in medias_pruebas.keys()):
            troponina_tni_array.append(medias_pruebas[6012])
        else:
            troponina_tni_array.append(-1)

        # Añadir media de pruebas de TROPONINA (TNI) al paciente
        if (6013 in medias_pruebas.keys()):
            troponina_tnt_array.append(medias_pruebas[6013])
        else:
            troponina_tnt_array.append(-1)

        # Añadir media de pruebas de ACIDO URICO al paciente
        """
        if (- in medias_pruebas.keys()):
            acido_urico_array.append(medias_pruebas[-])
        else:
            acido_urico_array.append(-1)
        """

        # Añadir media de pruebas de DIMERO D al paciente
        if (10989 in medias_pruebas.keys()):
            dimero_d_array.append(medias_pruebas[10989])
        else:
            dimero_d_array.append(-1)

        # Añadir media de pruebas de FIBRINOGENO al paciente
        if (10962 in medias_pruebas.keys()):
            fibrinogeno_array.append(medias_pruebas[10962])
        else:
            fibrinogeno_array.append(-1)

        # Añadir media de pruebas de ALDOSTERONA al paciente
        if (4497 in medias_pruebas.keys()):
            aldosterona_array.append(medias_pruebas[4497])
        else:
            aldosterona_array.append(-1)

        # Añadir media de pruebas de PCR al paciente
        """
        if (- in ultimas_pruebas.keys()):
            pcr_array.append(ultimas_pruebas[-])
        else:
            pcr_array.append(-1)
        """

        # Añadir media de pruebas de LEUCOCITOS al paciente
        if (2445 in medias_pruebas.keys()):
            leucocitos_array.append(medias_pruebas[2445])
        else:
            leucocitos_array.append(-1)

        # Añadir media de pruebas de PCT al paciente
        if (6011 in medias_pruebas.keys()):
            pct_array.append(medias_pruebas[6011])
        else:
            pct_array.append(-1)

        # Añadir media de pruebas de TSH al paciente
        if (5412 in medias_pruebas.keys()):
            tsh_array.append(medias_pruebas[5412])
        else:
            tsh_array.append(-1)

        # Añadir media de pruebas de T4L al paciente
        if (5391 in medias_pruebas.keys()):
            t4l_array.append(medias_pruebas[5391])
        else:
            t4l_array.append(-1)

    tabla_prafai['urea'] = urea_array                   # Añadir columna UREA a la tabla_prafai
    tabla_prafai['creatinina'] = creatinina_array       # Añadir columna CRATININA a la tabla_prafai
    tabla_prafai['albumina'] = albumina_array           # Añadir columna ALBUMINA a la tabla_prafai
    tabla_prafai['glucosa'] = glucosa_array             # Añadir columna GLUCOSA a la tabla_prafai
    tabla_prafai['hba1c'] = hba1c_array                 # Añadir columna HbA1c a la tabla_prafai
    tabla_prafai['potasio'] = potasio_array             # Añadir columna POTASIO a la tabla_prafai
    tabla_prafai['calcio'] = calcio_array               # Añadir columna CALCIO a la tabla_prafai
    tabla_prafai['hdl'] = hdl_array                     # Añadir columna HDL a la tabla_prafai
    tabla_prafai['no_hdl'] = no_hdl_array                     # Añadir columna No HDL a la tabla_prafai
    tabla_prafai['ldl'] = ldl_array                     # Añadir columna LDL a la tabla_prafai
    tabla_prafai['colesterol'] = colesterol_array       # Añadir columna COLESTEROL a la tabla_prafai
    tabla_prafai['ntprobnp'] = ntprobnp_array           # Añadir columna NT-proBNP a la tabla_prafai
    tabla_prafai['bnp'] = bnp_array           # Añadir columna BNP a la tabla_prafai
    tabla_prafai['troponina_tni'] = troponina_tni_array # Añadir columna TROPONINA (TNI) a la tabla_prafai
    tabla_prafai['troponina_tnt'] = troponina_tnt_array # Añadir columna TROPONINA (TNT) a la tabla_prafai
    #tabla_prafai['acido_urico'] = acido_urico_array     # Añadir columna ACIDO URICO a la tabla_prafai
    tabla_prafai['dimero_d'] = dimero_d_array           # Añadir columna DIMERO D a la tabla_prafai
    tabla_prafai['fibrinogeno'] = fibrinogeno_array     # Añadir columna FIBRINOGENO a la tabla_prafai
    tabla_prafai['aldosterona'] = aldosterona_array     # Añadir columna ALDOSTERONA a la tabla_prafai
    #tabla_prafai['pcr'] = pcr_array                     # Añadir columna PCR a la tabla_prafai
    tabla_prafai['leucocitos'] = leucocitos_array       # Añadir columna LEUCOCITOS a la tabla_prafai
    tabla_prafai['pct'] = pct_array                     # Añadir columna PCT a la tabla_prafai
    tabla_prafai['tsh'] = tsh_array                     # Añadir columna TSH a la tabla_prafai
    tabla_prafai['t4l'] = tsh_array                     # Añadir columna T4L a la tabla_prafai

    return tabla_prafai


def cargar_laboratorio_extra(laboratorio, tabla_prafai):
    # Columnas a crear (sodio...)
    urea_array = []
    creatinina_array = []
    sodio_array = []
    acido_urico_array = []
    pcr_array = []
    vsg_array = []
    tl3_array = []

    for index, row in tabla_prafai.iterrows():      # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']

        pruebas = {}            # Se almacenan TODAS las pruebas realizadas a ese paciente ANTES Y DESPUES DEL DIAGNOSTICO DE FA
        medias_pruebas = {}     # Se almacenan LAS MEDIAS de todas pruebas realizadas a ese paciente ANTES Y DESPUES DEL DIAGNOSTICO DE FA

        pruebas_paciente = laboratorio[laboratorio['Id Paciente'] == id]        # Se obtienen todas las pruebas realizadas a ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_prueba, row_prueba in pruebas_paciente.iterrows():        # Se va a recorrer cada prueba del paciente
            id_prueba = row_prueba['Id prueba']     # Se obtiene el id de esa prueba
            resultado_prueba = row_prueba['Resultado texto']    # Se obtiene el resultado de esa prueba
            try:    # Se intenta convertir el resultado a FLOAT, si no es posible no se tiene en cuenta
                resultado_prueba_float = float(resultado_prueba)
                if(resultado_prueba_float < 10000):     # EVITAR VALORES INCORRECTOS (LIMITE 10000)
                    if (id_prueba in pruebas.keys()):
                        # La prueba ya se le ha realizado mas veces al paciente
                        pruebas[id_prueba].append(resultado_prueba_float)  # Se añade el resultado de la prueba
                    else:
                        # Es la primera vez que se le realiza la prueba al paciente
                        pruebas[id_prueba] = [resultado_prueba_float]  # Se guarda su resultado
            except ValueError:
                pass
                #print("PRUEBA NO FLOAT: " + str(resultado_prueba))

        for prueba in pruebas.keys():    # Se recorren todas la pruebas del paciente ANTES DEL DIAGNOSTICO DE FA
            medias_pruebas[prueba] = round(np.mean(pruebas[prueba]),3)   # Se guarda la media de resultados de la prueba realizada al paciente

        # Añadir media de pruebas de UREA al paciente (+ VALOR ANTERIOR)
        urea_values = []
        for cod in medias_pruebas.keys():
            if ((cod == 173) | (cod == 208) |
            (cod == 5436) | (cod == 5438) |
            (cod == 8411) | (cod == 5440) |
            (cod == 6213) | (cod == 6212) |
            (cod == 2099) | (cod == 2100) |
            (cod == 8403) | (cod == 8404)):
                urea_values.append(medias_pruebas[cod])

        urea_anterior = row['urea']
        if (len(urea_values) > 0):
            urea_values.append(urea_anterior)
            urea_array.append(round(np.mean(urea_values), 3))
        else:
            urea_array.append(urea_anterior)

        # Añadir media de pruebas de Creatinina al paciente
        creatinina_nueva = -1
        if (6217 in medias_pruebas.keys()):
            creatinina_nueva = medias_pruebas[6217]

        creatinina_anterior = row['creatinina']
        if (creatinina_nueva != -1):
            creatinina_array.append(round(np.mean([creatinina_anterior, creatinina_nueva]), 3))
        else:
            creatinina_array.append(creatinina_anterior)

        # Añadir media de pruebas de SODIO al paciente
        sodio_values = []
        for cod in medias_pruebas.keys():
            if ((cod == 191) | (cod == 217) |
            (cod == 305) | (cod == 5226) |
            (cod == 8414) | (cod == 5230) |
            (cod == 6245) | (cod == 6244)):
                sodio_values.append(medias_pruebas[cod])
        if(len(sodio_values) > 0):
            sodio_array.append(round(np.mean(sodio_values),3))
        else:
            sodio_array.append(-1)

        # Añadir media de pruebas de ACIDO URICO al paciente
        """
        if (- in medias_pruebas.keys()):
            acido_urico_array.append(medias_pruebas[-])
        else:
            acido_urico_array.append(-1)
        """

        # Añadir media de pruebas de PCR al paciente
        """
        if (- in ultimas_pruebas.keys()):
            pcr_array.append(ultimas_pruebas[-])
        else:
            pcr_array.append(-1)
        """

        # Añadir media de pruebas de VSG al paciente
        if (4492 in medias_pruebas.keys()):
            vsg_array.append(medias_pruebas[4492])
        else:
            vsg_array.append(-1)

        # Añadir media de pruebas de TL3 al paciente
        if (5390 in medias_pruebas.keys()):
            tl3_array.append(medias_pruebas[5390])
        else:
            tl3_array.append(-1)

    tabla_prafai['urea'] = urea_array                   # Añadir columna UREA a la tabla_prafai
    tabla_prafai['creatinina'] = creatinina_array       # Añadir columna CREATININA a la tabla_prafai
    tabla_prafai['sodio'] = sodio_array                 # Añadir columna SODIO a la tabla_prafai
    #tabla_prafai['acido_urico'] = acido_urico_array    # Añadir columna ACIDO URICO a la tabla_prafai
    #tabla_prafai['pcr'] = pcr_array                    # Añadir columna PCR a la tabla_prafai
    tabla_prafai['vsg'] = vsg_array                     # Añadir columna VSG a la tabla_prafai
    tabla_prafai['tl3'] = tl3_array                     # Añadir columna TL3 a la tabla_prafai

    return tabla_prafai


def cargar_procedimientos_1(pruebas, tabla_prafai):
    # Columnas a crear (ecocardiograma, ecocardiograma_contraste y electrocardiograma)
    ecocardiograma_array = []
    ecocardiograma_contraste_array = []
    electrocardiograma_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        procedimientos = {}  # Se almacenan TODOS los procedimientos realizados a ese paciente ANTES DEL DIAGNOSTICO DE FA
        ultimos_procedimientos = {}  # Se almacenan LOS ÚLTIMOS procedimientos realizados a ese paciente ANTES DEL DIAGNOSTICO DE FA

        procedimientos_paciente = pruebas[pruebas['Id Paciente'] == id]  # Se obtienen todas los procedimientos realizados a ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_procedimiento, row_procedimiento in procedimientos_paciente.iterrows():  # Se va a recorrer cada procedimiento del paciente
            fecha_procedimiento = row_procedimiento['Fecha (cita)']  # Se obtiene la fecha de ese procedimiento en formato STRING
            if (fecha_procedimiento < fecha):  # Solo se tiene en cuenta el procedimiento si la fecha es anterior al diagnostico de FA
                cod_procedimiento = row_procedimiento['cod Prueba (cita)']  # Se obtiene el codigo de ese procedimiento
                if (cod_procedimiento in procedimientos.keys()):
                    # El procedimiento ya se le ha realizado mas veces al paciente
                    procedimientos[cod_procedimiento].append(fecha_procedimiento)  # Se añade la fecha de realizacion del procedimiento
                else:
                    # Es la primera vez que se le realiza el procedimiento al paciente
                    procedimientos[cod_procedimiento] = [fecha_procedimiento]  # Se guarda la fecha de realizacion del procedimiento


        for procedimiento in procedimientos.keys():  # Se recorren todos los procedimientos del paciente ANTES DEL DIAGNOSTICO DE FA
            ultima_fecha = np.max(procedimientos[procedimiento])  # Se obtiene la fecha del ultimo procedimiento realizado al paciente
            indice_ultima_fecha = procedimientos[procedimiento].index(ultima_fecha)
            ultimos_procedimientos[procedimiento] = procedimientos[procedimiento][indice_ultima_fecha]  # Se guarda la fecha del ultimo procedimiento ralizado al paciente

        # Añadir si se le ha realizado el ECOCARDIOGRAMA (ECOCARDIOGRAFIA) al paciente
        if ((90022 in ultimos_procedimientos.keys()) | (90159 in ultimos_procedimientos.keys()) | (90365 in ultimos_procedimientos.keys()) | (90021 in ultimos_procedimientos.keys())):
            ecocardiograma_array.append(1)
        else:
            ecocardiograma_array.append(0)

        # Añadir si se le ha realizado el ECOCARDIOGRAMA_CONTRASTE al paciente
        if (90138 in ultimos_procedimientos.keys()):
            ecocardiograma_contraste_array.append(1)
        else:
            ecocardiograma_contraste_array.append(0)

        # Añadir si se le ha realizado el ELECTROCARDIOGRAMA al paciente
        if (90023 in ultimos_procedimientos.keys()):
            electrocardiograma_array.append(1)
        else:
            electrocardiograma_array.append(0)

    tabla_prafai['ecocardiograma'] = ecocardiograma_array                       # Añadir columna ECOCARDIOGRAMA a la tabla_prafai
    tabla_prafai['ecocardiograma_contraste'] = ecocardiograma_contraste_array   # Añadir columna ECOCARDIOGRAMA_CONTRASTE a la tabla_prafai
    tabla_prafai['electrocardiograma'] = electrocardiograma_array               # Añadir columna ELECTROCARDIOGRAMA a la tabla_prafai

    return tabla_prafai


def cargar_procedimientos_2(primeros_outcome, tabla_prafai):
    # Columnas a crear (FEVI, diametro_AI y area_AI)
    fevi_array = []
    diametro_ai_array = []
    area_ai_array = []

    for index, row in primeros_outcome.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        fevi_paciente = row['FEVI T0']              # Se obtiene el valor de FEVI para ese paciente
        diametro_ai_paciente = row['AI (MM) T0']    # Se obtiene el valor de diamtro_AI para ese paciente
        area_ai_paciente = row['AI (cm2) T0']       # Se obtiene el valor de area_AI para ese paciente

        try:  # Se intenta convertir el resultado de FEVI a FLOAT, si no es posible no se tiene en cuenta
            fevi_paciente_float = float(fevi_paciente)
            fevi_array.append(fevi_paciente_float)  # Añadir valor de FEVI al paciente
        except ValueError:
            if ("-" in fevi_paciente) & (not "%" in fevi_paciente):    # En el caso de valores 20-30, calcular media (hay algun valor 40-50%)
                fevi_min = float(fevi_paciente.split("-")[0])
                fevi_max = float(fevi_paciente.split("-")[1])
                fevi_mean = np.mean([fevi_min, fevi_max])
                fevi_array.append(fevi_mean)  # Añadir valor de FEVI al paciente
            else:
                #print("FEVI NO FLOAT: " + str(fevi_paciente))
                fevi_array.append(-1)

        try:  # Se intenta convertir el resultado de DIAMETRO AI a FLOAT, si no es posible no se tiene en cuenta
            diametro_ai_paciente_float = float(diametro_ai_paciente)
            diametro_ai_array.append(diametro_ai_paciente_float)  # Añadir valor de diametro_AI al paciente
        except ValueError:
            if (("-" in diametro_ai_paciente) & (not "%" in diametro_ai_paciente)):     # En el caso de valores 20-30, calcular media (hay algun valor 40-50%)
                diametro_min = float(diametro_ai_paciente.split("-")[0])
                diametro_max = float(diametro_ai_paciente.split("-")[1])
                diametro_ai_mean = np.mean([diametro_min, diametro_max])
                diametro_ai_array.append(diametro_ai_mean)  # Añadir valor de diametro_AI al paciente
            else:
                #print("DIAMETRO AI NO FLOAT: " + str(diametro_ai_paciente))
                diametro_ai_array.append(-1)

        try:  # Se intenta convertir el resultado de AREA AI a FLOAT, si no es posible no se tiene en cuenta
            area_ai_paciente_float = float(area_ai_paciente)
            area_ai_array.append(area_ai_paciente_float)  # Añadir valor de area_AI al paciente
        except ValueError:
            if (("-" in area_ai_paciente) & (not "%" in area_ai_paciente)):     # En el caso de valores 20-30, calcular media (hay algun valor 40-50%)
                area_min = float(area_ai_paciente.split("-")[0])
                area_max = float(area_ai_paciente.split("-")[1])
                area_ai_mean = np.mean([area_min, area_max])
                area_ai_array.append(area_ai_mean)  # Añadir valor de area_AI al paciente
            else:
                #print("AREA AI NO FLOAT: " + str(area_ai_paciente))
                area_ai_array.append(-1)

    tabla_prafai['fevi'] = fevi_array                     # Añadir columna FEVI a la tabla_prafai
    tabla_prafai['diametro_ai'] = diametro_ai_array       # Añadir columna diametro_AI a la tabla_prafai
    tabla_prafai['area_ai'] = area_ai_array               # Añadir columna area_AI a la tabla_prafai

    # Se sustituyen los valores NaN a 'No hay información'
    tabla_prafai['fevi'] = tabla_prafai['fevi'].fillna(-1)
    tabla_prafai['diametro_ai'] = tabla_prafai['diametro_ai'].fillna(-1)
    tabla_prafai['area_ai'] = tabla_prafai['area_ai'].fillna(-1)

    return tabla_prafai


def cargar_intervenciones(procedimientos_quirurgicos, tabla_prafai):
    procedimientos_quirurgicos['Fecha (realización proc)'] = procedimientos_quirurgicos['Fecha (realización proc)'].fillna("Unknown")   # HAY ALGUNA FECHA VACIA --> Se transforma a UNKNOWN

    # Columnas a crear (cardioversion, ablacion y marcapasos_DAI)
    cardioversion_array = []
    ablacion_array = []
    marcapasos_dai_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        intervenciones = {}  # Se almacenan TODAS las intervenciones realizadas a ese paciente DESPUES DEL DIAGNOSTICO DE FA
        ultimas_intervenciones = {}  # Se almacenan LAS ÚLTIMAS intercenciones realizadas a ese paciente DESPUES DEL DIAGNOSTICO DE FA

        intervenciones_paciente = procedimientos_quirurgicos[procedimientos_quirurgicos['Id Paciente'] == id]  # Se obtienen todas las intervenciones realizadas a ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_intervencion, row_intervencion in intervenciones_paciente.iterrows():  # Se va a recorrer cada intervencion del paciente
            fecha_intervencion = row_intervencion['Fecha (realización proc)']  # Se obtiene la fecha de esa intervencion en formato STRING
            if(fecha_intervencion != 'Unknown'):    # HAY ALGUNA FECHA VACIA
                cod_intervencion = row_intervencion['cod Procedimiento']  # Se obtiene el codigo de esa intervencion
                if ((fecha_intervencion > fecha) |                         # Solo se tiene en cuenta la intervencion si la fecha es posterior al diagnostico de FA
                        (cod_intervencion in [3781, 3782, 3783, 53, 54, 3796])):    # En el caso de marcapasos o DAI da igual el momento de la intervencion (ANTERIOR O POSTERIOR AL DIAGNOSTICO FA)
                    if (cod_intervencion in intervenciones.keys()):
                        # La intervencion ya se le ha realizado mas veces al paciente
                        intervenciones[cod_intervencion].append(fecha_intervencion)  # Se añade la fecha de realizacion de la intervencion
                    else:
                        # Es la primera vez que se le realiza la intervencion al paciente
                        intervenciones[cod_intervencion] = [fecha_intervencion]  # Se guarda la fecha de realizacion de la intervencion

        for intervencion in intervenciones.keys():  # Se recorren todos las intervenciones del paciente DESPUES DEL DIAGNOSTICO DE FA
            ultima_fecha = np.max(intervenciones[intervencion])  # Se obtiene la fecha de la ultima intervencion realizada al paciente
            indice_ultima_fecha = intervenciones[intervencion].index(ultima_fecha)
            ultimas_intervenciones[intervencion] = intervenciones[intervencion][indice_ultima_fecha]  # Se guarda la fecha de la ultima intervencion realizada al paciente

        # Añadir si se le ha realizado la CARDIOVERSION al paciente
        if (9962 in ultimas_intervenciones.keys()):
            cardioversion_array.append(1)
        else:
            cardioversion_array.append(0)

        # Añadir si se le ha realizado la ABLACION al paciente
        if (3734 in ultimas_intervenciones.keys()):
            ablacion_array.append(1)
        else:
            ablacion_array.append(0)

        # Añadir si se le ha realizado la intervencion MARCAPASOS o DAI al paciente
        if ((3781 in ultimas_intervenciones.keys()) |
                (3782 in ultimas_intervenciones.keys()) |
                (3783 in ultimas_intervenciones.keys()) |
                (53 in ultimas_intervenciones.keys()) |
                (54 in ultimas_intervenciones.keys()) |
                (3796 in ultimas_intervenciones.keys())):
            marcapasos_dai_array.append(1)
        else:
            marcapasos_dai_array.append(0)


    tabla_prafai['cardioversion'] = cardioversion_array             # Añadir columna CARDIOVERSION a la tabla_prafai
    tabla_prafai['ablacion'] = ablacion_array                       # Añadir columna ALBLACION a la tabla_prafai
    tabla_prafai['marcapasos_dai'] = marcapasos_dai_array           # Añadir columna marcapasos_DAI a la tabla_prafai

    return tabla_prafai


def cargar_numero_ingresos(ingresos, tabla_prafai):
    # Columnas a crear (numero_ingresos)
    numero_ingresos_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']

        numero_ingresos = 0  # Se inicializa el numero de ingresos del paciente a 0

        ingresos_paciente = ingresos[ingresos['Id Paciente'] == id]  # Se obtienen todas los ingresos de ese paciente (SOLO HAY UNO POR PACIENTE)
        for index_ingresos, row_ingresos in ingresos_paciente.iterrows():  # Se va a recorrer cada ingreso del paciente
            numero_ingresos = row_ingresos['Ingresos']  # Se obtiene el numero de ingresos de ese paciente

        numero_ingresos_array.append(numero_ingresos)   # Añadir numero de ingresos del paciente


    tabla_prafai['numero_ingresos'] = numero_ingresos_array     # Añadir columna numero_ingresos a la tabla_prafai

    return tabla_prafai


def cargar_ultimo_ingreso(ultimo_ingreso, tabla_prafai):

    # Columnas a crear (etilogia_cardiologica, numero_dias_desde_ingreso_hasta_evento y numero_dias_ingresado)
    etilogia_cardiologica_array = []
    numero_dias_desde_ingreso_hasta_evento_array = []
    numero_dias_ingresado_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        ingreso_paciente = ultimo_ingreso[ultimo_ingreso['Id Paciente'] == id]  # Se obtiene el ultimo ingreso del paciente (puede ser ANTES O DESPUES DEL DIAGNOSTICO DE FA)

        # Inicializar las variables que van a recoger los tres datos necesarios
        cod_ingreso = []
        diferencia_dias = -1
        numero_dias_ingresado = -1

        for index_ingreso, row_ingreso in ingreso_paciente.iterrows():  # Se va a recorrer cada ingreso (PUEDE HABER MAS DE UNO POR PACIENTE, PERO MISMA FECHA Y ESTANCIA)
            fecha_ingreso = row_ingreso['Fecha (ingreso)']  # Se obtiene la fecha del ultimo ingreso en formato STRING
            if (fecha_ingreso < fecha):    # Solo se tiene en cuenta el ingreso si la fecha es anterior al diagnostico de FA
                cod_ingreso_aux = str(row_ingreso['cod Diagnóstico CIE9'])  # Se obtiene el codigo de la razon del ingreso
                if(cod_ingreso_aux.isdigit()):  # FILTRO PARA EL CODIGO (ALGUNOS CONTIENEN LETRAS)
                    cod_ingreso.append(int(cod_ingreso_aux))
                diferencia_dias = (fecha - fecha_ingreso).days  # Se calcula la diferencia de dias desde el ingreso hasta el evento de FA
                numero_dias_ingresado = row_ingreso['Estancia (días)']   # Se obtiene el numero de dias ingresado

        # Añadir si es un CIE9 cardiologico (390 - 459)
        variable_CIE9_cardiologico = 0
        for cod in cod_ingreso:
            if (390 <= cod <= 459):
                variable_CIE9_cardiologico = 1
                break
        etilogia_cardiologica_array.append(variable_CIE9_cardiologico)

        # Añadir numero de dias desde el ingreso
        numero_dias_desde_ingreso_hasta_evento_array.append(diferencia_dias)

        # Añadir numero de dias ingresado
        numero_dias_ingresado_array.append(numero_dias_ingresado)

    tabla_prafai['etilogia_cardiologica'] = etilogia_cardiologica_array                                     # Añadir columna etilogia_cardiologica a la tabla_prafai
    tabla_prafai['numero_dias_desde_ingreso_hasta_evento'] = numero_dias_desde_ingreso_hasta_evento_array   # Añadir columna numero_dias_desde_ingreso_hasta_evento a la tabla_prafai
    tabla_prafai['numero_dias_ingresado'] = numero_dias_ingresado_array                                     # Añadir columna numero_dias_ingresado a la tabla_prafai

    return tabla_prafai


def cargar_diagnosticos(diagnosticos_fich, tabla_prafai):
    # Columnas a crear (depresion, alcohol...)
    depresion_array = []
    alcohol_array = []
    drogodependencia_array = []
    ansiedad_array = []
    demencia_array = []
    insuficiencia_renal_array = []
    menopausia_array = []
    osteoporosis_array = []
    diabetes_tipo1_array = []
    diabetes_tipo2_array = []
    dislipidemia_array = []
    hipercolesterolemia_array = []
    fibrilacion_palpitacion_array = []
    flutter_array = []
    insuficiencia_cardiaca_array = []
    fumador_array = []
    sahos_array = []
    hipertiroidismo_array = []
    sindrome_metabolico_array = []
    hipertension_arterial_array = []
    cardiopatia_isquemica_array = []
    ictus_array = []
    miocardiopatia_array = []
    otras_arritmias_psicogena_ritmo_bigeminal_array = []
    bloqueos_rama_array = []
    bloqueo_auriculoventricular_array = []
    bradicardia_array = []
    contracciones_prematuras_ectopia_extrasistolica_array = []
    posoperatoria_array = []
    sinusal_coronaria_array = []
    valvula_mitral_reumaticas_array = []
    otras_valvulopatias_array = []
    valvulopatia_mitral_congenita_array = []
    arteriopatia_periferica = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        diagnosticos = set()  # Se almacenan TODAS los diagnosticos de ese paciente ANTES DEL DIAGNOSTICO DE FA

        diagnosticos_paciente = diagnosticos_fich[diagnosticos_fich['Id Paciente'] == id]  # Se obtienen todos los diagnosticos de ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_diagnostico, row_diagnostico in diagnosticos_paciente.iterrows():  # Se va a recorrer cada diagnostico del paciente
            fecha_diagnostico = row_diagnostico['Fecha (inicio diag)']  # Se obtiene la fecha de ese diagnostico en formato STRING
            if (fecha_diagnostico < fecha):  # Solo se tiene en cuenta el diagnostico si la fecha es anterior al diagnostico de FA
                cod_diagnostico = row_diagnostico['cod CIE9']  # Se obtiene el CIE9 de ese diagnostico
                diagnosticos.add(cod_diagnostico)   # Se añade el codigo CIE9 del diagnostico

        # Añadir si se le a diagnosticado DEPRESION
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('296')):
                depresion_array.append(1)
                encontrado = True
                break
        if not encontrado:
            depresion_array.append(0)

        # Añadir si se le a diagnosticado ALCOHOL
        encontrado = False
        for cie9 in diagnosticos:
            if ((str(cie9).startswith('29181')) | (str(cie9).startswith('3039')) | (str(cie9).startswith('3050'))):
                alcohol_array.append(1)
                encontrado = True
                break
        if not encontrado:
            alcohol_array.append(0)

        # Añadir si se le a diagnosticado DROGODEPENDENCIA
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('304')):
                drogodependencia_array.append(1)
                encontrado = True
                break
        if not encontrado:
            drogodependencia_array.append(0)

        # Añadir si se le a diagnosticado ANSIEDAD
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('300')):
                ansiedad_array.append(1)
                encontrado = True
                break
        if not encontrado:
            ansiedad_array.append(0)

        # Añadir si se le a diagnosticado DEMENCIA
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('2942')):
                demencia_array.append(1)
                encontrado = True
                break
        if not encontrado:
            demencia_array.append(0)

        # Añadir si se le a diagnosticado INSUFICIENCIA RENAL
        if ('5939' in diagnosticos):
            insuficiencia_renal_array.append(1)
        else:
            insuficiencia_renal_array.append(0)

        # Añadir si se le a diagnosticado MENOPAUSIA
        if ('6272' in diagnosticos):
            menopausia_array.append(1)
        else:
            menopausia_array.append(0)

        # Añadir si se le a diagnosticado OSTEOPOROSIS
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('7330')):
                osteoporosis_array.append(1)
                encontrado = True
                break
        if not encontrado:
            osteoporosis_array.append(0)

        # Añadir si se le a diagnosticado DIABETES TIPO1
        if (('25001' in diagnosticos) | ('25003' in diagnosticos)):
            diabetes_tipo1_array.append(1)
        else:
            diabetes_tipo1_array.append(0)

        # Añadir si se le a diagnosticado DIABETES TIPO2
        if (('25000' in diagnosticos) | ('25002' in diagnosticos)):
            diabetes_tipo2_array.append(1)
        else:
            diabetes_tipo2_array.append(0)

        # Añadir si se le a diagnosticado DISLIPIDEMIA
        if ('2724' in diagnosticos):
            dislipidemia_array.append(1)
        else:
            dislipidemia_array.append(0)

        # Añadir si se le a diagnosticado HIPERCOLESTEROLEMIA
        if ('2720' in diagnosticos):
            hipercolesterolemia_array.append(1)
        else:
            hipercolesterolemia_array.append(0)

        # Añadir si se le a diagnosticado FIBRILACION-PALPITACION
        if ('42731' in diagnosticos):
            fibrilacion_palpitacion_array.append(1)
        else:
            fibrilacion_palpitacion_array.append(0)

        # Añadir si se le a diagnosticado FLUTTER
        if ('42732' in diagnosticos):
            flutter_array.append(1)
        else:
            flutter_array.append(0)

        # Añadir si se le a diagnosticado INSUFICIENCIA CARDIACA
        if (('4289' in diagnosticos) | ('4289 R' in diagnosticos)):
            insuficiencia_cardiaca_array.append(1)
        else:
            insuficiencia_cardiaca_array.append(0)

        # Añadir si se le a diagnosticado FUMADOR
        if (('98984' in diagnosticos) | ('3051' in diagnosticos)):
            fumador_array.append(1)
        else:
            fumador_array.append(0)

        # Añadir si se le a diagnosticado SAHOS
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('7805')):
                sahos_array.append(1)
                encontrado = True
                break
        if not encontrado:
            sahos_array.append(0)

        # Añadir si se le a diagnosticado HIPERTIROIDISMO
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('242')):
                hipertiroidismo_array.append(1)
                encontrado = True
                break
        if not encontrado:
            hipertiroidismo_array.append(0)

        # Añadir si se le a diagnosticado SINDROME METABOLICO
        if ('2777' in diagnosticos):
            sindrome_metabolico_array.append(1)
        else:
            sindrome_metabolico_array.append(0)

        # Añadir si se le a diagnosticado HIPERTENSION ARTERIAL
        if (('40400' in diagnosticos) |
                ('40410' in diagnosticos) |
                ('40490' in diagnosticos) |
                ('4010' in diagnosticos) |
                ('40300' in diagnosticos) |
                ('40310' in diagnosticos) |
                ('40390' in diagnosticos) |
                ('40501' in diagnosticos) |
                ('40511' in diagnosticos) |
                ('40591' in diagnosticos) |
                ('40509' in diagnosticos) |
                ('40519' in diagnosticos) |
                ('40599' in diagnosticos)):
            hipertension_arterial_array.append(1)
        else:
            hipertension_arterial_array.append(0)

        # Añadir si se le a diagnosticado CARDIOPATIA ISQUEMICA
        encontrado = False
        for cie9 in diagnosticos:
            if ((str(cie9).startswith('410')) |
                    (str(cie9).startswith('411')) |
                    (str(cie9).startswith('412')) |
                    (str(cie9).startswith('413')) |
                    (str(cie9).startswith('414'))):
                cardiopatia_isquemica_array.append(1)
                encontrado = True
                break
        if not encontrado:
            cardiopatia_isquemica_array.append(0)

        # Añadir si se le a diagnosticado ICTUS
        if ('43491' in diagnosticos):
            ictus_array.append(1)
        else:
            ictus_array.append(0)

        # Añadir si se le a diagnosticado MIOCARDIOPATIA
        if ('4254' in diagnosticos):
            miocardiopatia_array.append(1)
        else:
            miocardiopatia_array.append(0)

        # Añadir si se le a diagnosticado OTRAS ARRITMIAS
        if (('4279' in diagnosticos) |
                ('3062' in diagnosticos) |
                ('42789' in diagnosticos)):
            otras_arritmias_psicogena_ritmo_bigeminal_array.append(1)
        else:
            otras_arritmias_psicogena_ritmo_bigeminal_array.append(0)

        # Añadir si se le a diagnosticado BLOQUEOS RAMA
        if (('42651' in diagnosticos) | ('42652' in diagnosticos)):
            bloqueos_rama_array.append(1)
        else:
            bloqueos_rama_array.append(0)

        # Añadir si se le a diagnosticado BLOQUEO AURICULOVENTRICULAR
        encontrado = False
        for cie9 in diagnosticos:
            if ((str(cie9).startswith('4260')) | (str(cie9).startswith('4261'))):
                bloqueo_auriculoventricular_array.append(1)
                encontrado = True
                break
        if not encontrado:
            bloqueo_auriculoventricular_array.append(0)

        # Añadir si se le a diagnosticado BRADICARDIA
        if ('42789' in diagnosticos):
            bradicardia_array.append(1)
        else:
            bradicardia_array.append(0)

        # Añadir si se le a diagnosticado CONTRACCIONES PREMATURAS
        if (('42760' in diagnosticos) | ('42789' in diagnosticos)):
            contracciones_prematuras_ectopia_extrasistolica_array.append(1)
        else:
            contracciones_prematuras_ectopia_extrasistolica_array.append(0)

        # Añadir si se le a diagnosticado POSOPERATORIA
        if ('9971' in diagnosticos):
            posoperatoria_array.append(1)
        else:
            posoperatoria_array.append(0)

        # Añadir si se le a diagnosticado SINUSAL CORONARIA
        if ('42789' in diagnosticos):
            sinusal_coronaria_array.append(1)
        else:
            sinusal_coronaria_array.append(0)

        # Añadir si se le a diagnosticado VALVULA MITRAL
        encontrado = False
        for cie9 in diagnosticos:
            if (str(cie9).startswith('394')):
                valvula_mitral_reumaticas_array.append(1)
                encontrado = True
                break
        if not encontrado:
            valvula_mitral_reumaticas_array.append(0)

        # Añadir si se le a diagnosticado OTRAS VALVULOPATIAS
        if (('4240' in diagnosticos) |
                ('4241' in diagnosticos) |
                ('4242' in diagnosticos) |
                ('4243' in diagnosticos)):
            otras_valvulopatias_array.append(1)
        else:
            otras_valvulopatias_array.append(0)

        # Añadir si se le a diagnosticado VALVULOPATIA MITRAL CONGENITA
        if (('7245' in diagnosticos) | ('7246' in diagnosticos)):
            valvulopatia_mitral_congenita_array.append(1)
        else:
            valvulopatia_mitral_congenita_array.append(0)

        # Añadir si se le a diagnosticado ARTERIOPATIA PERIFERICA
        if ('4439' in diagnosticos):
            arteriopatia_periferica.append(1)
        else:
            arteriopatia_periferica.append(0)

    tabla_prafai['depresion'] = depresion_array                                 # Añadir columna DEPRESION a la tabla_prafai
    tabla_prafai['alcohol'] = alcohol_array                                     # Añadir columna ALCOHOL a la tabla_prafai
    tabla_prafai['drogodependencia'] = drogodependencia_array                   # Añadir columna DROGODEPENDENCIA a la tabla_prafai
    tabla_prafai['ansiedad'] = ansiedad_array                                   # Añadir columna ANSIEDAD a la tabla_prafai
    tabla_prafai['demencia'] = demencia_array                                   # Añadir columna DEMENCIA a la tabla_prafai
    tabla_prafai['insuficiencia_renal'] = insuficiencia_renal_array             # Añadir columna INSUFICIENCIA RENAL a la tabla_prafai
    tabla_prafai['menopausia'] = menopausia_array                               # Añadir columna MENOPAUSIA a la tabla_prafai
    tabla_prafai['osteoporosis'] = osteoporosis_array                           # Añadir columna OSTEOPOROSIS a la tabla_prafai
    tabla_prafai['diabetes_tipo1'] = diabetes_tipo1_array                       # Añadir columna DIABETES TIPO1 a la tabla_prafai
    tabla_prafai['diabetes_tipo2'] = diabetes_tipo2_array                       # Añadir columna DIABETES TIPO2 a la tabla_prafai
    tabla_prafai['dislipidemia'] = dislipidemia_array                           # Añadir columna DISLIPIDEMIA a la tabla_prafai
    tabla_prafai['hipercolesterolemia'] = hipercolesterolemia_array             # Añadir columna HIPERCOLESTEROLEMIA a la tabla_prafai
    tabla_prafai['fibrilacion_palpitacion'] = fibrilacion_palpitacion_array     # Añadir columna FRIBRILACION-PALPITACION a la tabla_prafai
    tabla_prafai['flutter'] = flutter_array                                     # Añadir columna FLUTTER a la tabla_prafai
    tabla_prafai['insuficiencia_cardiaca'] = insuficiencia_cardiaca_array       # Añadir columna INSUFICIENCIA CARDIACA a la tabla_prafai
    tabla_prafai['fumador'] = fumador_array                                     # Añadir columna FUMADOR a la tabla_prafai
    tabla_prafai['sahos'] = sahos_array                                         # Añadir columna SAHOS a la tabla_prafai
    tabla_prafai['hipertiroidismo'] = hipertiroidismo_array                     # Añadir columna HIPERTIROIDISMO a la tabla_prafai
    tabla_prafai['sindrome_metabolico'] = sindrome_metabolico_array             # Añadir columna SINDROME METABOLICO a la tabla_prafai
    tabla_prafai['hipertension_arterial'] = hipertension_arterial_array         # Añadir columna HIPERTENSION ARTERIAL a la tabla_prafai
    tabla_prafai['cardiopatia_isquemica'] = cardiopatia_isquemica_array         # Añadir columna CARDIOPATIA ISQUEMICA a la tabla_prafai
    tabla_prafai['ictus'] = ictus_array                                         # Añadir columna ICTUS a la tabla_prafai
    tabla_prafai['miocardiopatia'] = miocardiopatia_array                       # Añadir columna MIOCARDIOPATIA a la tabla_prafai
    tabla_prafai['otras_arritmias_psicogena_ritmo_bigeminal'] = otras_arritmias_psicogena_ritmo_bigeminal_array  # Añadir columna OTRAS ARRITMIAS a la tabla_prafai
    tabla_prafai['bloqueos_rama'] = bloqueos_rama_array                         # Añadir columna BLOQUEOS RAMA a la tabla_prafai
    tabla_prafai['bloqueo_auriculoventricular'] = bloqueo_auriculoventricular_array  # Añadir columna BLOQUEO AURICULOVENTRICULAR a la tabla_prafai
    tabla_prafai['bradicardia'] = bradicardia_array                             # Añadir columna BRADICARDIA a la tabla_prafai
    tabla_prafai['contracciones_prematuras_ectopia_extrasistolica'] = contracciones_prematuras_ectopia_extrasistolica_array  # Añadir columna CONTRACCIONES PREMATURAS a la tabla_prafai
    tabla_prafai['posoperatoria'] = posoperatoria_array                         # Añadir columna POSOPERATORIA a la tabla_prafai
    tabla_prafai['sinusal_coronaria'] = sinusal_coronaria_array                 # Añadir columna SINUSAL CORONARIA a la tabla_prafai
    tabla_prafai['valvula_mitral_reumaticas'] = valvula_mitral_reumaticas_array # Añadir columna VALVULA MITRAL a la tabla_prafai
    tabla_prafai['otras_valvulopatias'] = otras_valvulopatias_array             # Añadir columna OTRAS VALVULOPATIAS a la tabla_prafai
    tabla_prafai['valvulopatia_mitral_congenita'] = valvulopatia_mitral_congenita_array  # Añadir columna VALVULOPATIA MITRAL CONGENITA a la tabla_prafai
    tabla_prafai['arteriopatia_periferica'] = arteriopatia_periferica           # Añadir columna ARTERIOPATIA PERIFERICA a la tabla_prafai

    return tabla_prafai


def cargar_epoc(epoc_fich, tabla_prafai):
    # Columnas a crear (epoc)
    epoc_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        epoc_bool = 0
        diagnosticos_paciente = epoc_fich[epoc_fich['Identificador de Paciente'] == id]  # Se obtienen todos los diagnosticos de ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_diagnostico, row_diagnostico in diagnosticos_paciente.iterrows():  # Se va a recorrer cada diagnostico del paciente
            cod_diagnostico = row_diagnostico['Código CIE']  # Se obtiene el CIE9 de ese diagnostico
            if (cod_diagnostico == 496): # Solo se tiene en cuenta el diagnostico di tiene CIE9 496
                fecha_diagnostico = row_diagnostico['Día (Fecha)']  # Se obtiene la fecha de ese diagnostico
                if (fecha_diagnostico < fecha):  # Solo se tiene en cuenta el diagnostico si la fecha es anterior al diagnostico de FA
                    epoc_bool = 1   # Se ha encontrado diagnostico EPOC
                    break
        epoc_array.append(epoc_bool)    # Se añade diagnostico de EPOC al paciente

    tabla_prafai['epoc'] = epoc_array      # Añadir columna EPOC a la tabla_prafai

    return tabla_prafai


def cargar_datos_paciente(datos_de_paciente, tabla_prafai):
    # Columnas a crear (genero, edad, pensionista y residencia)
    genero_array = []
    edad_array = []
    pensionista_array = []
    residencia_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        datos = {}  # Se almacenan TODOS los datos de ese paciente

        datos_paciente = datos_de_paciente[datos_de_paciente['Identificador de Paciente'] == id]  # Se obtienen todos los datos de ese paciente
        for index_datos, row_datos in datos_paciente.iterrows():  # Se va a recorrer cada fila de datos del paciente (SOLO HAY UNA FILA POR PACIENTE)
            genero = row_datos['Sexo']      # Se obtiene el genero de ese paciente
            if(genero == 'Mujer'):
                datos['genero'] = 0         # Mujer --> 0
            else:
                datos['genero'] = 1         # Hombre --> 1
            fecha_nacimiento = row_datos['Fecha de Nacimiento']     # Se obtiene la fecha de naciemiento de ese paciente en formato STRING
            diff_anos = dateutil.relativedelta.relativedelta(fecha, fecha_nacimiento).years     # Se calcula la edad del paciente en el momento del diagnostico
            datos['edad'] = diff_anos    # Se calcula la diferencia de la fecha de nacimiento con el EPOCH TIME
            pensionista = row_datos['Trabajador/pensionista de la ficha TIS']   # Se obtiene si el paciente es pensionista (1) o no (0)
            if (pensionista == 'Pensionista'):
                datos['pensionista'] = 1
            elif (pensionista == 'Trabajador'):
                datos['pensionista'] = 0
            else:
                if(diff_anos >= 65):    # Si no hay informacion se toma como pensionista si es mayor de 65 anos
                    datos['pensionista'] = 1
                else:
                    datos['pensionista'] = 0
            residenciado = row_datos['Residenciado (S/N)']      # Se obtiene si el paciente esta residenciado (1) o no (0)
            if(residenciado == 'Si'):
                datos['residenciado'] = 1
            elif(residenciado == 'No'):
                datos['residenciado'] = 0

        # Añadir GENERO del paciente
        genero_array.append(datos['genero'])

        # Añadir EDAD del paciente
        edad_array.append(datos['edad'])

        # Añadir si el paciente es PENSIONISTA
        pensionista_array.append(datos['pensionista'])

        # Añadir si el paciente es RESIDENCIADO
        residencia_array.append(datos['residenciado'])

    tabla_prafai['genero'] = genero_array               # Añadir columna GENERO a la tabla_prafai
    tabla_prafai['edad'] = edad_array                   # Añadir columna EDAD a la tabla_prafai
    tabla_prafai['pensionista'] = pensionista_array     # Añadir columna PENSIONISTA a la tabla_prafai
    tabla_prafai['residenciado'] = residencia_array     # Añadir columna RESIDENCIADO a la tabla_prafai

    return tabla_prafai


def cargar_talla_peso_imc(talla_peso_imc, tabla_prafai):
    # Columnas a crear (talla, IMC y peso)
    talla_array = []
    imc_array = []
    peso_array = []

    for index, row in tabla_prafai.iterrows():      # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']

        datos = {7: [], 8:[], 9:[]}                          # Se almacenan TODAS los datos de ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO FA)
        datos_medias = {}                   # Se almacena la MEDIA de los datos de ese paciente

        datos_paciente = talla_peso_imc[talla_peso_imc['Identificador de Paciente'] == id]        # Se obtienen todas los datos ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_datos, row_datos in datos_paciente.iterrows():        # Se va a recorrer cada prueba del paciente
            cod_dato = row_datos['Código del DBP']  # Se obtiene el codigo de ese dato
            valor_dato = float(row_datos['Valor DBP'])  # Se obtiene el valor de ese dato
            if(valor_dato != 0):    # Si es 0 se descarta
                if(cod_dato == 8):     # Correccion datos talla
                    if(1.0 <= valor_dato <= 1.99):
                        valor_dato = valor_dato * 100
                        datos[8].append(valor_dato)
                    elif((not "." in str(valor_dato)) & (len(str(valor_dato)) == 6)):
                        valor_dato = valor_dato / 1000
                        datos[8].append(valor_dato)
                    elif((not "." in str(valor_dato)) & (len(str(valor_dato)) == 4)):
                        valor_dato = valor_dato / 10
                        datos[8].append(valor_dato)
                    elif(100 <= valor_dato <= 200):
                        datos[8].append(valor_dato)
                elif(cod_dato == 9):    # Correccion datos peso
                    if((not "." in str(valor_dato)) & ((len(str(valor_dato)) == 6) | (len(str(valor_dato)) == 5))):
                        datos[9].append(valor_dato)
                    elif(10 <= valor_dato <= 200):
                        valor_dato = valor_dato * 1000
                        datos[9].append(valor_dato)
                elif (cod_dato == 7):  # Correccion datos IMC
                    if(len(str(float(valor_dato))) == 5):
                        valor_dato = valor_dato * 1000
                    elif(len(str(float(valor_dato))) == 4):
                        valor_dato = valor_dato * 100
                    if(10 <= valor_dato <= 70):
                        datos[7].append(valor_dato)

        for cod in datos:   # Por cada dato (Talla/Peso/IMC) se calculara la media
            if(not datos[cod]):
                media = round(-1,2)
            else:
                media = round(np.mean(datos[cod]),2)
            datos_medias[cod] = media

        # Añadir media de TALLA al paciente
        talla_array.append(datos_medias[8])

        # Añadir media de IMC al paciente
        imc_array.append(datos_medias[7])

        # Añadir media de PESO al paciente
        peso_array.append(datos_medias[9])

    tabla_prafai['talla'] = talla_array             # Añadir columna TALLA a la tabla_prafai
    tabla_prafai['imc'] = imc_array                 # Añadir columna IMC a la tabla_prafai
    tabla_prafai['peso'] = peso_array               # Añadir columna PESO a la tabla_prafai

    return tabla_prafai

def recalcular_imc_posibles(tabla_prafai):
    # Columna a recalcular (IMC)
    imc_array = []

    for index, row in tabla_prafai.iterrows():      # Se va a recorrer cada paciente diagnosticado con FA
        imc = row['imc']

        if(imc == -1):      # Si el paciente tenia missing value en el IMC se va a intentar recalcular
            peso = row['peso']      # Se obtiene el peso del paciente
            talla = row['talla']    # Se obtiene la talla del paciente
            if((peso != -1) & (talla != -1)):       # Si el peso y la talla no son missing values se recalcula el IMC
                imc_recalculado = (peso/1000) / pow((talla/100),2)      # IMC = peso(kg) / talla(m^2)
                if(10 <= imc_recalculado <= 70):    # Si el IMC esta entre [10, 70] se acepta
                    imc = round(imc_recalculado,3)

        # Clasificar IMC mediante los siguientes criterios:
        #   0: [10,16] --> Desnutricion severa
        #   1: (16, 18.5) --> Desnutricion moderada
        #   2: [18.5, 22] --> Bajo peso
        #   3: (22, 25) --> Peso normal
        #   4: [25, 30) --> Sobrepeso
        #   5: [30, 35) --> Obesidad tipo I
        #   6: [35, 40) --> Obesidad tipo II
        #   7: [40, 70] --> Obesidad tipo III
        if(10 <= imc <= 16):
            imc = 0
        elif(16 < imc < 18.5):
            imc = 1
        elif (18.5 <= imc <= 22):
            imc = 2
        elif (22 < imc < 25):
            imc = 3
        elif (25 <= imc < 30):
            imc = 4
        elif (30 <= imc < 35):
            imc = 5
        elif (35 <= imc < 40):
            imc = 6
        elif (40 <= imc <= 70):
            imc = 7

        # Añadir IMC recalculado al paciente
        imc_array.append(imc)

    tabla_prafai['imc'] = imc_array  # Sustituir columna IMC de la tabla_prafai

    return tabla_prafai


def cargar_tratamientos(hft, tabla_prafai):
    hft['Día (Fecha)'] = hft['Día (Fecha)'].fillna("Unknown")  # HAY ALGUNA FECHA VACIA --> Se transforma a UNKNOWN

    # Columnas a crear (B01A, N02BA01...)
    b01a_array = []
    n02ba01_array = []
    a02bc_array = []
    c03_array = []
    g03a_array = []
    a10_array = []
    n06a_array = []
    n05a_array = []
    n05b_array = []
    c01_array = []
    c01b_array = []
    c02_array = []
    c04_array = []
    c07_array = []
    c08_array = []
    c09_array = []
    c10_array = []
    polimedicacion_array = []

    for index, row in tabla_prafai.iterrows():  # Se va a recorrer cada paciente diagnosticado con FA
        id = row['id']
        fecha = row['fecha']

        tratamientos = set()  # Se almacenan TODAS los tratamientos de ese paciente DESPUES DEL DIAGNOSTICO DE FA (Y UN AÑO DE ANTELACION)

        tratamientos_paciente = hft[hft['Identificador de Paciente'] == id]  # Se obtienen todos los tratamientos de ese paciente (ANTES Y DESPUES DEL DIAGNOSTICO DE FA)
        for index_tratamiento, row_tratamiento in tratamientos_paciente.iterrows():  # Se va a recorrer cada tratamiento del paciente
            fecha_tratamiento = row_tratamiento['Día (Fecha)']  # Se obtiene la fecha de ese tratamiento en formato STRING
            if (fecha_tratamiento != 'Unknown'):  # HAY ALGUNA FECHA VACIA
                if (fecha_tratamiento > (fecha - dateutil.relativedelta.relativedelta(years=1))):  # Solo se tiene en cuenta el tratamiento si la fecha es posterior al diagnostico de FA (o con un año de antelacion)
                    atc_tratamiento = row_tratamiento['ATC']  # Se obtiene el codigo de ese tratamiento
                    tratamientos.add(atc_tratamiento)  # Se añade el codigo ATC del tratamiento

        polimedicacion_contador = 0     # Contador del numero de tratamientos del paciente

        # Añadir si se le ha tratado con B01A
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('B01A')):
                b01a_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            b01a_array.append(0)

        # Añadir si se le ha tratado con N02BA01
        if ('N02BA01' in tratamientos):
            n02ba01_array.append(1)
            polimedicacion_contador += 1
        else:
            n02ba01_array.append(0)

        # Añadir si se le ha tratado con A02BC
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('A02BC')):
                a02bc_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            a02bc_array.append(0)

        # Añadir si se le ha tratado con C03
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C03')):
                c03_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c03_array.append(0)

        # Añadir si se le ha tratado con G03A
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('G03A')):
                g03a_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            g03a_array.append(0)

        # Añadir si se le ha tratado con A10
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('A10')):
                a10_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            a10_array.append(0)

        # Añadir si se le ha tratado con N06A
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('N06A')):
                n06a_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            n06a_array.append(0)

        # Añadir si se le ha tratado con N05A
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('N05A')):
                n05a_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            n05a_array.append(0)

        # Añadir si se le ha tratado con N05B
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('N05B')):
                n05b_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            n05b_array.append(0)

        # Añadir si se le ha tratado con C01
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C01')):
                c01_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c01_array.append(0)

        # Añadir si se le ha tratado con C01B
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C01B')):
                c01b_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c01b_array.append(0)

        # Añadir si se le ha tratado con C02
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C02')):
                c02_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c02_array.append(0)

        # Añadir si se le ha tratado con C04
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C04')):
                c04_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c04_array.append(0)

        # Añadir si se le ha tratado con C07
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C07')):
                c07_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c07_array.append(0)

        # Añadir si se le ha tratado con C08
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C08')):
                c08_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c08_array.append(0)

        # Añadir si se le ha tratado con C09
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C09')):
                c09_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c09_array.append(0)

        # Añadir si se le ha tratado con C10
        encontrado = False
        for atc in tratamientos:
            if (str(atc).startswith('C10')):
                c10_array.append(1)
                encontrado = True
                polimedicacion_contador += 1
                break
        if not encontrado:
            c10_array.append(0)

        # Añadir si el paciente tiene POLIMEDICACION
        if (polimedicacion_contador >= 5):
            polimedicacion_array.append(1)
        else:
            polimedicacion_array.append(0)


    tabla_prafai['b01a'] = b01a_array           # Añadir columna B01A a la tabla_prafai
    tabla_prafai['n02ba01'] = n02ba01_array     # Añadir columna N02BA01 a la tabla_prafai
    tabla_prafai['a02bc'] = a02bc_array         # Añadir columna A02BC a la tabla_prafai
    tabla_prafai['c03'] = c03_array             # Añadir columna C03 a la tabla_prafai
    tabla_prafai['g03a'] = g03a_array           # Añadir columna G03A a la tabla_prafai
    tabla_prafai['a10'] = a10_array             # Añadir columna A10 a la tabla_prafai
    tabla_prafai['n06a'] = n06a_array           # Añadir columna N06A a la tabla_prafai
    tabla_prafai['n05a'] = n05a_array           # Añadir columna N05A a la tabla_prafai
    tabla_prafai['n05b'] = n05b_array           # Añadir columna N05B a la tabla_prafai
    tabla_prafai['c01'] = c01_array             # Añadir columna C01 a la tabla_prafai
    tabla_prafai['c01b'] = c01b_array           # Añadir columna C01B a la tabla_prafai
    tabla_prafai['c02'] = c02_array             # Añadir columna C01B a la tabla_prafai
    tabla_prafai['c04'] = c04_array             # Añadir columna C04 a la tabla_prafai
    tabla_prafai['c07'] = c07_array             # Añadir columna C07 a la tabla_prafai
    tabla_prafai['c08'] = c08_array             # Añadir columna C08 a la tabla_prafai
    tabla_prafai['c09'] = c09_array             # Añadir columna C09 a la tabla_prafai
    tabla_prafai['c10'] = c10_array             # Añadir columna C10 a la tabla_prafai
    tabla_prafai['polimedicacion'] = polimedicacion_array  # Añadir columna POLIMEDICACION a la tabla_prafai

    return tabla_prafai


def cargar_clase(primeros_outcome, tabla_prafai):
    # Columna a añadir (sigue_fa)
    sigue_fa_array = []

    for index, row in primeros_outcome.iterrows():      # Se va a recorrer cada paciente diagnosticado con FA
        sigue_fa = row['FA PERM 2A?']       # Se obtiene la variable que indica si el paciente permanece en FA
        if((sigue_fa != 0) & (sigue_fa != 1)):      # Si hay missing value se asigna -1
            sigue_fa = -1

        # Añadir variable sigue_fa al paciente
        sigue_fa_array.append(sigue_fa)

    tabla_prafai['sigue_fa'] = sigue_fa_array   # Añadir columna sigue_fa de la tabla_prafai

    return tabla_prafai


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create the PRAFAI dataset. Input folder with following files is needed: DatosDePaciente.xlsx, Diagnosticos.xlsx, EPOCgrupo1.xlsx, EPOCgrupo2.xlsx, HTF_1_1.xlsx, HFT_1_2.xlsx, HTF_1_3.xlsx, HTF_2_1.xlsx, HTF_2_2.xlsx, HTF_2_3.xlsx, Laboratorio.xlsx, LaboratorioGrupo1.xlsx, LaboratorioGrupo2.xlsx, NumeroIngresos.xlsx, Outcomes.xlsx, ProcedimientosQuirurgicos.xlsx, Pruebas.xlsx, TallaPesoIMC.xlsx, UltimoIngreso.xlsx.')
    parser.add_argument("input_dir", help="Path to directory with input data.")
    parser.add_argument("-o", "--output_dir", help="Path to directory for the created dataset. Default option: current directory.", default=os.getcwd())
    args = vars(parser.parse_args())
    input_dir = args['input_dir']
    output_dir = args['output_dir']

    print('Input directory: ' + str(input_dir))
    print('Output directory: ' + str(output_dir))

    print('\nReading input files...')
    ficheros = leer_ficheros()
    print("Loading 'Outcomes.xlsx'...")
    dataset = cargar_datos_FA(ficheros['outcomes'])
    print("Loading 'Laboratorio.xlsx'...")
    dataset = cargar_laboratorio(ficheros['laboratorio'], dataset)
    print("Loading 'LaboratorioGrupo1.xlsx' and 'LaboratorioGrupo2.xlsx'...")
    dataset = cargar_laboratorio_extra(ficheros['laboratorio_extra'], dataset)
    print("Loading 'Pruebas.xlsx'...")
    dataset = cargar_procedimientos_1(ficheros['pruebas'], dataset)
    dataset = cargar_procedimientos_2(ficheros['outcomes'], dataset)
    print("Loading 'ProcedimientosQuirurgicos.xlsx'...")
    dataset = cargar_intervenciones(ficheros['procedimientos_quirurgicos'], dataset)
    print("Loading 'NumeroIngresos.xlsx'...")
    dataset = cargar_numero_ingresos(ficheros['numero_ingresos'], dataset)
    print("Loading 'UltimoIngreso.xlsx'...")
    dataset = cargar_ultimo_ingreso(ficheros['ultimo_ingreso'], dataset)
    print("Loading 'Diagnosticos.xlsx'...")
    dataset = cargar_diagnosticos(ficheros['diagnosticos'], dataset)
    print("Loading 'EPOCgrupo1.xlsx' and 'EPOCgrupo2.xlsx'...")
    dataset = cargar_epoc(ficheros['epoc'], dataset)
    print("Loading 'DatosDePaciente.xlsx'...")
    dataset = cargar_datos_paciente(ficheros['datos_de_paciente'], dataset)
    print("Loading 'TallaPesoIMC.xlsx'...")
    dataset = cargar_talla_peso_imc(ficheros['talla_peso_imc'], dataset)
    dataset = recalcular_imc_posibles(dataset)
    print("Loading 'HFT_1_1.xlsx', 'HFT_1_2.xlsx', 'HFT_1_3.xlsx', 'HFT_2_1.xlsx', 'HFT_2_2.xlsx' and 'HFT_2_3.xlsx'...")
    dataset = cargar_tratamientos(ficheros['hft'], dataset)
    dataset = cargar_clase(ficheros['outcomes'], dataset)

    # Reemplazar todos los valores -1 por NaN
    dataset = dataset.replace(-1, np.nan)

    print('\nSaving PRAFAI dataset to path: ' + str(os.path.join(output_dir,"dataset.csv")))
    dataset.to_csv(os.path.join(output_dir,"dataset.csv"), index=False, sep=';')



