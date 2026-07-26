import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETRI DI BASE 
# ==========================================
V_s = 18.0          # Tensione di ingresso [V]
f_s = 200e3         # Frequenza di switching [Hz]
L_r = 6.1e-6        # Induttore risonante [H]
C_r1 = 104e-9       # Condensatore risonante parallelo [F]
C_r2 = 62e-9        # Condensatore risonante serie [F]
C_s = 460e-12         # Capacità parassita (Snubber/Coss) degli interruttori [F]

N = 24              # Numero di LED
V_D = 3.3           # Tensione di ginocchio del singolo LED [V]
I_o_target = 0.5    # Corrente nominale LED [A]
V_s_min = 18.0      # Tensione minima di ingresso [V]
V_Do = 0.7          # Caduta di tensione dei diodi di uscita [V]
eta = 0.90          # Efficienza stimata del convertitore (90%)
L_B = 80e-6         # Induttanza di Boost [H] 
P_o = N * V_D * I_o_target # Potenza di uscita nominale [W]

# Vettori per l'asse X dei grafici
D_array = np.linspace(0.1, 0.85, 500)       # Duty Cycle da 0.1 a 0.85
wn_array = np.linspace(0.4, 1.8, 500)       # Frequenza normalizzata

# ==========================================
# 2. DEFINIZIONE DELLE 11 EQUAZIONI
# ==========================================

def eq_vo2(D):
    return V_s / (1 - D)

def eq_vab1(D):
    return (V_s / (np.pi * (1 - D))) * np.sqrt(2 * (1 - np.cos(2 * np.pi * D)))

def eq_wo():
    return 1 / np.sqrt(L_r * C_r1)

def eq_fi(D):
    y = np.sin(2 * np.pi * D)
    x = 1 - np.cos(2 * np.pi * D)
    return np.arctan2(y, x)

def eq_Cn():
    return C_r2 / C_r1

def eq_omegan():
    w_s = 2 * np.pi * f_s
    return w_s / eq_wo()

def eq_Z0():
    return np.sqrt(L_r / C_r1)

def eq_Re(D):
    # Usa eq_Io(D) invece del valore fisso I_o_target per precisione dinamica
    term1 = (2 / np.pi**2) * (N * V_D / eq_Io(D)) 
    term2 = 1 - (V_s / (N * V_D * (1 - D)))
    return term1 * term2
   # term1 = (2 / np.pi**2) * (N * V_D / I_o_target)
   # term2 = 1 - (V_s / (N * V_D * (1 - D)))
  #  return term1 * term2

def eq_Qe(D):
    """
    Fattore di merito Qe = Z0 / Re
    
    NOTA FISICA E MATEMATICA SULL'ASINTOTO:
    Per D attorno a 0.773, la resistenza Re si annulla e Qe tende all'infinito.
    Matematicamente questo avviene quando: 1 - Vs / (N * VD * (1 - D)) = 0.
    Fisicamente corrisponde al punto in cui la tensione DC generata dal Boost 
    eguaglia esattamente la tensione richiesta dalla stringa LED (circa 79.2V).
    Per D > 0.773, l'ipotesi della "prima armonica" su cui si basa questo 
    modello matematico perde validità.
    """
    return eq_Z0() / eq_Re(D)

def eq_M(wn, Qe):
    Cn = eq_Cn()
    num = 1j * Cn * wn
    den = Qe * (1 - wn**2 * (1 + Cn)) + 1j * Cn * wn * (1 - wn**2)
    return np.abs(num / den)

def eq_H(wn, Qe):
    Cn = eq_Cn()
    num = 1j * Cn * wn
    den = 1 - wn**2 * (1 + Cn) + 1j * Cn * Qe * wn * (1 - wn**2)
    return np.abs(num / den)

def eq_Io(D):
    # Formula: Io = (1/π²) * (Vs/Z0) * (√(2*(1 - cos(2πD))) / (1 - D))
    term1 = 1 / (np.pi**2)
    term2 = V_s / eq_Z0()
    term3 = np.sqrt(2 * (1 - np.cos(2 * np.pi * D))) / (1 - D)
    return term1 * term2 * term3

def eq_Vo_min(D):
    # Formula limite: Vo_min = (Vs_min / (1 - D)) + 2*V_Do
    return (V_s_min / (1 - D)) + 2 * V_Do

def eq_Vo1(D):
    # Formula: VDo1 = VDo2 = Vo1 = N*VD - Vs/(1-D)
    return (N * V_D) - (V_s / (1 - D))

def eq_ILB_max(D):
    # Formula: ILB_max = (Po / (eta * Vs)) + (4 * D * Vs) / (pi^2 * LB * fs)
    term1 = P_o / (eta * V_s)
    term2 = (4 * D * V_s) / (np.pi**2 * L_B * f_s)
    return term1 + term2

def eq_alpha(D):
    # Formula: alpha = arctan(Qe * ((1 + Cn) / Cn))
    Cn = eq_Cn()
    return np.arctan(eq_Qe(D) * ((1 + Cn) / Cn))

def eq_ILr_amp(D):
    # Formula: Ampiezza di ILr = (V_AB1 / Z0) * sqrt(1/Qe^2 + (1+Cn)^2/Cn^2)
    # Calcoliamo solo l'ampiezza perché il tempo 't' non è sul nostro asse X
    Cn = eq_Cn()
    Qe = eq_Qe(D)
    V_AB1 = eq_vab1(D)
    Z0 = eq_Z0()
    
    # Gestiamo l'errore di divisione per zero quando Qe tende all'infinito
    with np.errstate(divide='ignore', invalid='ignore'):
        term_sqrt = np.sqrt((1 / Qe**2) + ((1 + Cn)**2 / Cn**2)) # <-- Corretto qui!
    
    return (V_AB1 / Z0) * term_sqrt

def eq_IDo_max(D):
    # Formula: IDo1_max = IDo2_max = pi * Io
    return np.pi * eq_Io(D)

def eq_IDo_DC(D):
    # Formula: IDo1_DC = IDo2_DC = Io
    return eq_Io(D)

def eq_ILB_t7(D):
    # Corrente induttore Boost all'istante t7
    term1 = P_o / (eta * V_s)
    term2 = (D * V_s) / (2 * L_B * f_s)
    return term1 - term2

def eq_ZVS_energia_capacitiva(D):
    # Termine di Sinistra (LHS): Energia richiesta per scaricare le capacità
    return C_s * (V_s / (1 - D))**2

def eq_ZVS_energia_induttiva(D):
    # Termine di Destra (RHS): Energia induttiva disponibile
    # NOTA: Usiamo eq_ILr_amp(D) come approssimazione di ILr(t7) in assenza della formula esatta
    I_Lr_t7 = eq_ILr_amp(D) 
    I_LB_t7 = eq_ILB_t7(D)
    return 0.5 * L_r * (I_Lr_t7**2 - I_LB_t7**2)

def eq_Vo_carico(D):
    # Parametri del modello SPICE del LED e costanti fisiche
    N_led = 24
    n_emiss = 3.48
    Is = 1e-15
    Rs = 0.5
    Vt = 0.02585  # Tensione termica a 27°C (300K)
    
    # Prende la corrente teorica imposta dal convertitore LCC
    I_out = eq_Io(D)
    
    # Calcola la tensione di un singolo LED e moltiplica per la stringa
    V_singolo_led = (n_emiss * Vt * np.log(I_out / Is)) + (Rs * I_out)
    return N_led * V_singolo_led

# ==========================================
# 3. LOGICA DI PLOT INTERATTIVA
# ==========================================
def plot_equation(scelta):
    plt.figure(figsize=(10, 6))
    
    if scelta == '1':
        plt.plot(D_array, eq_vo2(D_array), 'b-', linewidth=2)
        plt.title("Tensione vo2 vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("vo2 [V]")

        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        Vo2_points = eq_vo2(D_points)
        plt.plot(D_points, Vo2_points, 'ko', label='Punti Teorici')

        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Vo2_points):
            plt.annotate(f"{v:.1f}V", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()

    elif scelta == '2':
        plt.plot(D_array, eq_vab1(D_array), 'g-', linewidth=2)
        plt.title("Ampiezza prima armonica vab1 vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Ampiezza [V]")
        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        A_points = eq_vab1(D_points)
        plt.plot(D_points, A_points, 'ko', label='Punti Teorici')

        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, A_points):
            plt.annotate(f"{v:.1f}V", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
        
    elif scelta == '3':
        val = eq_wo()
        print(f"\n=> Valore costante calcolato wo: {val:.2f} rad/s")
        plt.axhline(val, color='c', linewidth=2)
        plt.title(f"Frequenza di risonanza wo = {val:.2f} rad/s")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("wo [rad/s]")
        
    elif scelta == '4':
        plt.plot(D_array, np.degrees(eq_fi(D_array)), 'r-', linewidth=2)
        plt.title("Sfasamento fi vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Fase [Gradi]")
        
    elif scelta == '5':
        Qe_list = [0.1, 0.5, 1, 2, 5]
        for Qe in Qe_list:
            plt.plot(wn_array, eq_M(wn_array, Qe), label=f'Qe = {Qe}')
        plt.title(f"Guadagno di Tensione |M| (Cn={eq_Cn():.2f})")
        plt.xlabel("Frequenza Normalizzata (omegan)")
        plt.ylabel("|M|")
        plt.axvline(1.0, color='k', linestyle='--', alpha=0.5)
        plt.legend()
        
    elif scelta == '6':
        Qe_list = [0.1, 0.5, 1, 2, 5]
        for Qe in Qe_list:
            plt.plot(wn_array, eq_H(wn_array, Qe), label=f'Qe = {Qe}')
        plt.title(f"Guadagno di Corrente |H| (Cn={eq_Cn():.2f})")
        plt.xlabel("Frequenza Normalizzata (omegan)")
        plt.ylabel("|H|")
        plt.plot(1.0, eq_H(1.0, 1.0), 'ro')
        plt.annotate('Load-Independent', xy=(1.02, eq_H(1.0, 1.0)), xytext=(1.2, 1.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        plt.legend()
        
    elif scelta == '7':
        val = eq_omegan()
        print(f"\n=> Valore costante calcolato omegan: {val:.4f}")
        plt.axhline(val, color='orange', linewidth=2)
        plt.title(f"Frequenza Normalizzata Operativa omegan = {val:.4f}")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("omegan")

    elif scelta == '8':
        val = eq_Cn()
        print(f"\n=> Valore costante calcolato Cn: {val:.4f}")
        plt.axhline(val, color='purple', linewidth=2)
        plt.title(f"Rapporto Condensatori Cn = {val:.4f}")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Cn")

    elif scelta == '9':
        val = eq_Z0()
        print(f"\n=> Valore costante calcolato Z0: {val:.2f} Ohm")
        plt.axhline(val, color='brown', linewidth=2)
        plt.title(f"Impedenza Caratteristica Z0 = {val:.2f} Ohm")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Z0 [Ohm]")

    elif scelta == '10':
        plt.plot(D_array, eq_Qe(D_array), 'k-', linewidth=2)
        plt.title("Fattore di merito Qe vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Qe")
        
        # Calcolo e tracciamento dell'asintoto
        D_critico = 1 - (V_s / (N * V_D))
        plt.axvline(D_critico, color='r', linestyle='--', alpha=0.7, label=f'Punto critico (D={D_critico:.3f})')
        
        # Mostra la legenda per far comparire l'etichetta del punto critico
        plt.legend()

    elif scelta == '11':
        plt.plot(D_array, eq_Re(D_array), 'm-', linewidth=2)
        plt.title("Resistenza AC Equivalente Re vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Re [Ohm]")
        
        # Tracciamento l'asse dello zero per far capire dove la resistenza si annulla
        plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.7)
        
        # Aggiunta la linea verticale del punto critico (come fatto per Qe)
        D_critico = 1 - (V_s / (N * V_D))
        plt.axvline(D_critico, color='r', linestyle='--', alpha=0.5, label=f'Punto critico (D={D_critico:.3f})')
        
        # limitato l'asse Y per tagliare la discesa inutile e centrare l'attenzione
        plt.ylim(-5, 30) 
        plt.legend()
    
    elif scelta == '12':
        plt.plot(D_array, eq_Io(D_array), 'c-', linewidth=2)
        plt.title("Corrente di uscita Io vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Io [A]")
        
        # Linea orizzontale per vedere il target (0.5A)
        plt.axhline(I_o_target, color='r', linestyle='--', alpha=0.5, label=f'Target Io={I_o_target}A')
        plt.legend()

        # Evidenziati i punti esatti per D=0.3, 0.4, 0.5 sulla curva teorica
        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        Io_points = eq_Io(D_points)
        plt.plot(D_points, Io_points, 'ko', label='Punti Teorici')
        
        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Io_points):
            plt.annotate(f"{v:.2f}A", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()

    elif scelta == '13':
        plt.plot(D_array, eq_Vo_min(D_array), 'y-', linewidth=2)
        plt.title("Soglia minima Tensione Vo vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Vo_min [V]")

    elif scelta == '14':
        plt.plot(D_array, eq_Vo1(D_array), 'g-', linewidth=2)
        plt.title("Tensione $V_{o1}$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("V_o1 [V]")
        
        # Aggiunta la linea dello zero perché questa tensione diventa negativa a D alti
        plt.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.7)
        plt.legend(["$V_{o1}$"])


                # Evidenziati i punti esatti per D=0.3, 0.4, 0.5 sulla curva teorica
        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        Io_points = eq_Vo1(D_points)
        plt.plot(D_points, Io_points, 'ko', label='Punti Teorici')
        
        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Io_points):
            plt.annotate(f"{v:.2f}V", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()

    elif scelta == '15':
        plt.plot(D_array, eq_ILB_max(D_array), 'b-', linewidth=2)
        plt.title("Corrente Max Induttore Boost $I_{LB\_max}$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Corrente [A]")
        plt.grid(True)

        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        Io_points = eq_ILB_max(D_points)
        plt.plot(D_points, Io_points, 'ko', label='Punti Teorici')
        
        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Io_points):
            plt.annotate(f"{v:.2f}A", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()

    elif scelta == '16':
        plt.plot(D_array, np.degrees(eq_alpha(D_array)), 'r-', linewidth=2)
        plt.title("Angolo $\\alpha$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("$\\alpha$ [Gradi]")
        
        # Aggiungiamo l'asintoto visto che alpha dipende da Qe
        D_critico = 1 - (V_s / (N * V_D))
        plt.axvline(D_critico, color='k', linestyle='--', alpha=0.5, label=f'Punto critico (D={D_critico:.3f})')
        plt.legend()

    elif scelta == '17':
        plt.plot(D_array, eq_ILr_amp(D_array), 'b-', linewidth=2)
        plt.title("Ampiezza Corrente Induttore Risonante $I_{Lr\_amp}$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Ampiezza $I_{Lr}$ [A]")
                
        D_critico = 1 - (V_s / (N * V_D))
        plt.axvline(D_critico, color='r', linestyle='--', alpha=0.5, label=f'Punto critico (D={D_critico:.3f})')
        plt.legend()

        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        Io_points = eq_ILr_amp(D_points)
        plt.plot(D_points, Io_points, 'ko', label='Punti Teorici')
        
        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Io_points):
            plt.annotate(f"{v:.2f}A", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()

    elif scelta == '18':
        plt.plot(D_array, eq_IDo_max(D_array), 'g-', linewidth=2)
        plt.title("Corrente di picco Diodi Uscita $I_{Do\_max}$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Corrente [A]")

        D_points = np.array([0.3, 0.345, 0.4, 0.5])
        Io_points = eq_IDo_max(D_points)
        plt.plot(D_points, Io_points, 'ko', label='Punti Teorici')
        
        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Io_points):
            plt.annotate(f"{v:.2f}A", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()
    elif scelta == '19':
        plt.plot(D_array, eq_IDo_DC(D_array), 'm-', linewidth=2)
        plt.title("Corrente media Diodi Uscita $I_{Do\_DC}$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Corrente [A]")
        
        # Aggiunta il target per riferimento
        plt.axhline(I_o_target, color='r', linestyle='--', alpha=0.5, label=f'Target Io={I_o_target}A')
        plt.legend()

    elif scelta == '20':
        plt.plot(D_array, eq_ILB_t7(D_array), 'c-', linewidth=2)
        plt.title("Corrente Induttore Boost a t7: $I_{LB(t7)}$ vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Corrente [A]")

    elif scelta == '21':
        # Calcolo le energie e le converto in uJ per una lettura migliore
        E_cap = eq_ZVS_energia_capacitiva(D_array) * 1e6
        E_ind = eq_ZVS_energia_induttiva(D_array) * 1e6
        
        plt.plot(D_array, E_ind, 'g-', linewidth=2, label="Energia Induttiva Disponibile (RHS)")
        plt.plot(D_array, E_cap, 'r--', linewidth=2, label="Energia Capacitiva da scaricare (LHS)")
        
        plt.title("Condizione ZVS (Soft Switching) vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Energia [$\mu$J]")
        
        # Coloro l'area in cui si ha Soft Switching (RHS > LHS)
        plt.fill_between(D_array, E_ind, E_cap, where=(E_ind > E_cap), color='green', alpha=0.1, label='Zona ZVS Garantita')
        
        # Linea del punto critico
        D_critico = 1 - (V_s / (N * V_D))
        plt.axvline(D_critico, color='k', linestyle=':', alpha=0.5, label=f'Punto critico (D={D_critico:.3f})')
        
        plt.ylim(0, np.nanmax(E_cap[D_array < 0.75]) * 3) # Limito la Y per non farla esplodere
        plt.legend()

    elif scelta == '22':
        # Creo una griglia 2D di Duty Cycle (D) e Tensione (Vs)
        # Esattamente come i limiti degli assi nel paper
        D_vals = np.linspace(0.1, 0.9, 300)
        Vs_vals = np.linspace(4, 20, 300)
        D_grid, Vs_grid = np.meshgrid(D_vals, Vs_vals)

        # Ricalcolo le variabili dipendenti da Vs sulla griglia 2D
        # Usa il Cn fisso derivato dai condensatori (circa 0.6)
        Cn_val = eq_Cn() 
        Z0_val = eq_Z0()

        # Ricalcolo V_AB1 sulla griglia
        V_AB1_grid = (Vs_grid / (np.pi * (1 - D_grid))) * np.sqrt(2 * (1 - np.cos(2 * np.pi * D_grid)))

        # Ricalcolo Re sulla griglia
        term1_Re = (2 / np.pi**2) * (N * V_D / I_o_target)
        term2_Re = 1 - (Vs_grid / (N * V_D * (1 - D_grid)))
        Re_grid = term1_Re * term2_Re

        # Ricalcolo Qe e termine radice per ILr_amp ignorando le divisioni per zero matematiche
        with np.errstate(divide='ignore', invalid='ignore'):
            Qe_grid = Z0_val / Re_grid
            term_sqrt = np.sqrt((1 / Qe_grid**2) + ((1 + Cn_val)**2 / Cn_val**2))

        # Correnti sulla griglia
        I_Lr_amp_grid = (V_AB1_grid / Z0_val) * term_sqrt
        I_LB_t7_grid = (P_o / (eta * Vs_grid)) - ((D_grid * Vs_grid) / (2 * L_B * f_s))

        # Calcolo delle due energie (LHS e RHS) sulla griglia
        E_cap_grid = C_s * (Vs_grid / (1 - D_grid))**2
        E_ind_grid = 0.5 * L_r * (I_Lr_amp_grid**2 - I_LB_t7_grid**2)

        # La condizione ZVS si avvera quando l'energia induttiva è maggiore di quella capacitiva
        # Sottraggo per trovare il "margine". Se è > 0, siamo in ZVS.
        ZVS_margin = E_ind_grid - E_cap_grid

        # Creazione del plot stile paper
        plt.figure(figsize=(10, 6))
        
        # contourf colora le aree. Coloro tutto ciò che è maggiore di 0.
        plt.contourf(D_grid, Vs_grid, ZVS_margin, levels=[0, np.inf], colors=['#00c0c0'], alpha=0.9)
        
        # contour traccia la linea tratteggiata di confine (boundary) esattamente dove LHS = RHS
        plt.contour(D_grid, Vs_grid, ZVS_margin, levels=[0], colors=['blue'], linestyles=['dashed'], linewidths=2)

        plt.title(f"ZVS Region Map (Replicazione Paper con $C_n$={Cn_val:.2f})")
        plt.xlabel("Duty Cycle ($D$)")
        plt.ylabel("Tensione di Ingresso $V_s$ [V]")

        # Aggiunta l'etichetta testuale al centro della zona
        plt.text(0.5, 12, 'ZVS Region', fontsize=18, fontweight='bold', 
                 bbox=dict(facecolor='papayawhip', edgecolor='black', pad=10), ha='center')

        plt.xlim(0.1, 0.9)
        plt.ylim(4, 20)
        
        # Formattazione grafica
        plt.gca().tick_params(labelsize=12)
        plt.show()

    elif scelta == '23':
        plt.plot(D_array, eq_Vo_carico(D_array), 'm-', linewidth=2, label="Vo Teorica (Modello Diodo)")
        plt.title("Tensione di uscita sul Carico (Vo) vs Duty Cycle")
        plt.xlabel("Duty Cycle (D)")
        plt.ylabel("Tensione Vo [V]")
        
        # Aggiunte le linee orizzontali dei valori misurati in LTspice per un confronto visivo
        plt.axhline(77.8, color='b', linestyle='--', alpha=0.5, label='LTspice D=0.3 (~77.8V)')
        plt.axhline(80.8, color='r', linestyle='--', alpha=0.5, label='LTspice D=0.4 (~80.8V)')
        plt.axhline(83.6, color='c', linestyle='--', alpha=0.5, label='LTspice D=0.5 (~83.6V)')
        
        # Evidenziati i punti esatti per D=0.3, 0.4, 0.5 sulla curva teorica
        D_points = np.array([0.3, 0.4, 0.5])
        Vo_points = eq_Vo_carico(D_points)
        plt.plot(D_points, Vo_points, 'ko', label='Punti Teorici')
        
        # Annotazioni testuali per leggere i valori teorici esatti sul grafico
        for d, v in zip(D_points, Vo_points):
            plt.annotate(f"{v:.1f}V", (d, v), textcoords="offset points", xytext=(-15, 10), ha='center')
            
        plt.legend()

    else:
        print("\nErrore: Scelta non valida.")
        plt.close()
        return

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n" + "="*50)
        print(" SIMULATORE EQUAZIONI LCC IN ORDINE")
        print("="*50)
        print(" 1  - vo2 (Tensione Bus DC)")
        print(" 2  - vab1 (Ampiezza prima armonica)")
        print(" 3  - wo (Frequenza di risonanza - Costante)")
        print(" 4  - fi (Sfasamento)")
        print(" 5  - M (Guadagno Tensione vs omegan)")
        print(" 6  - H (Guadagno Corrente vs omegan)")
        print(" 7  - omegan (Frequenza normalizzata - Costante)")
        print(" 8  - Cn (Rapporto capacità - Costante)")
        print(" 9  - Z0 (Impedenza caratteristica - Costante)")
        print(" 10 - Qe (Fattore di merito)")
        print(" 11 - Re (Resistenza Equivalente)")
        print(" 12 - Io (Corrente di uscita)")
        print(" 13 - Vo_min (Soglia minima Tensione di uscita)")
        print(" 14 - Vo1 (Tensioni VDo1, VDo2)")
        print(" 15 - ILB_max (Corrente Max Induttore Boost)")
        print(" 16 - alpha (Angolo di fase alfa)")
        print(" 17 - ILr_amp (Ampiezza Corrente Induttore Risonante)")
        print(" 18 - IDo_max (Corrente picco Diodi)")
        print(" 19 - IDo_DC (Corrente media Diodi)")
        print(" 20 - ILB_t7 (Corrente Boost a t7)")
        print(" 21 - Condizione ZVS (Verifica Soft Switching)")
        print(" 22 - zvs")
        print(" 23 - vo")
        print(" q  - Esci")
        
        scelta = input("\nScegli la variabile da tracciare (1-23) o 'q' per uscire: ").strip().lower()
        
        if scelta == 'q':
            print("Chiusura del programma...")
            break
            
        plot_equation(scelta)

if __name__ == "__main__":
    main()