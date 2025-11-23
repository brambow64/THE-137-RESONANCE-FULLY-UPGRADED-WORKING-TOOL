import numpy as np
from scipy.ndimage import convolve
from scipy.fft import fftn
import itertools

# FUNDAMENTELE NATUURCONSTANTEN
PHI = (1 + np.sqrt(5)) / 2                    # PHI
ALPHA_INV = 137.035999084                     # α^-1
SCHUMANN_FUNDAMENTAL = 7.83                   # S 7.83
QUANTUM_HARMONICS = [3.0, 6.0, 9.0]          # 369
K_BOLTZMANN = 1.380649e-23                    # Kb
PLANCK_REDUCED = 1.054571817e-34              # h_bar
GEOMAGNETIC_PULSATIONS = [0.1, 0.2]           # G Puls

class NineLayerQuantumSystem:
    """⚛️ QCS: 9-Lagen Quantum Coherentie Systeem (Compacte Versie)"""
    
    def __init__(self, N=137, temperature=300.0, delta_t=1e-3):
        self.N = N
        self.dt = delta_t
        self.T = temperature
        self.age = 0
        self._initialize_layers()
        
    def _initialize_layers(self):
        """Initialisatie: N, T, h_bar -> CE_field"""
        thermal_freq = np.sqrt(K_BOLTZMANN * self.T / PLANCK_REDUCED)
        shape = (self.N, self.N, self.N)
        self.CE_field = np.tanh(np.random.normal(0, thermal_freq * self.dt, shape))
        self.GE_kernel3 = self._create_geometric_kernel(3) # 3k
        self.GE_kernel2 = self._create_geometric_kernel(2) # 2k
        self.history = {f'layer_{i}': [] for i in range(1, 10)}
        
    def _create_geometric_kernel(self, size):
        """L2: Geometrische kernel (inverse kwadraat)"""
        kernel = np.ones((size, size, size), np.float32)
        center = size // 2
        for i, j, k in itertools.product(range(size), repeat=3):
            if (i, j, k) != (center, center, center):
                r = np.sqrt((i-center)**2 + (j-center)**2 + (k-center)**2)
                kernel[i,j,k] = 1.0 / (r**2 + 1e-6)
        return kernel / np.sum(kernel)
    
    def execute_full_cycle(self):
        """Hoofdcyclus: CE → GE → PH → NS → FX → CW → C9 → DX → UN"""
        time_s = self.age * self.dt
        
        # L1: CE (Temporeel Fundament)
        CE_output = self._L1_CE(time_s)
        
        # L2: GE (Ruimtelijke Structuur)
        GE_output = self._L2_GE()
        
        # L3: PH (Harmonische Verhouding)
        PH_output = self._L3_PH(GE_output)
        
        # L4: NS (Frequentie Injectie)
        NS_output = self._L4_NS(PH_output, time_s)
        
        # L5: FX (Ruimtelijke Frequenties)
        FX_output = self._L5_FX(NS_output)
        
        # L6: CW (Structuurvorming/Evolutie)
        CW_output = self._L6_CW(NS_output, FX_output)
        
        # L7: C9 (Globale Orde Meting)
        C9_output = self._L7_C9(NS_output, FX_output, CW_output, time_s)
        
        # L8: DX (Diagnostiek)
        DX_output = self._L8_DX(CE_output, GE_output, PH_output, NS_output, 
                                FX_output, CW_output, C9_output, time_s)
        
        # L9: UN (Fysieke Manifestatie)
        UN_output = self._L9_UN(C9_output, DX_output, time_s)
        
        self.age += 1
        return UN_output

    # ========== COMPACTE LAGEN 1 t/m 7 ==========
    
    def _L1_CE(self, time_s):
        """L1: CE (1Hz, 369 subH)"""
        core_clock = np.sin(2 * np.pi * 1.0 * time_s)
        subharmonic_369 = sum(np.sin(2 * np.pi * f * time_s) for f in QUANTUM_HARMONICS) / 3.0
        CE_data = {'time': time_s, 'core_clock': core_clock, 'subharmonic': subharmonic_369, 'dt': self.dt}
        self.history['layer_1'].append(CE_data)
        return CE_data
    
    def _L2_GE(self):
        """L2: GE (n2, n3, tor3D)"""
        n3 = convolve(self.CE_field, self.GE_kernel3, mode='wrap')
        n2 = convolve(self.CE_field, self.GE_kernel2, mode='wrap')
        GE_data = {'n3': n3, 'n2': n2, 'topology': 'tor3D'}
        self.history['layer_2'].append(GE_data)
        return GE_data
    
    def _L3_PH(self, GE_output):
        """L3: PH (PHI, phiF = (n3 + P·n2) / (1+P))"""
        n3, n2 = GE_output['n3'], GE_output['n2']
        phi_weighted = (n3 + PHI * n2) / (1 + PHI)
        PH_data = {'phiF': phi_weighted, 'PHI': PHI}
        self.history['layer_3'].append(PH_data)
        return PH_data
    
    def _L4_NS(self, PH_output, time_s):
        """L4: NS (1/α N, S 7.83, 369 QP, G Puls)"""
        phi_field = PH_output['phiF']
        # 1/α N (Kwantumruis)
        noise_mask = (np.random.random(phi_field.shape) < 1/ALPHA_INV)
        noise_field = noise_mask * np.sign(phi_field) * 0.05
        # Resonanties (S 7.83, 369 QP, G Puls)
        schumann = np.sin(2 * np.pi * SCHUMANN_FUNDAMENTAL * time_s) * 0.1
        quantum_pulse = sum(np.sin(2 * np.pi * f * time_s) for f in QUANTUM_HARMONICS) / 3.0
        geomagnetic = sum(np.sin(2 * np.pi * f * time_s * 0.1) for f in GEOMAGNETIC_PULSATIONS) * 0.02
        
        resonance_modulation = 1.0 + 0.1 * (schumann + quantum_pulse + geomagnetic)
        modulated_field = phi_field * resonance_modulation + noise_field
        
        NS_data = {'mF': modulated_field, 'QP': quantum_pulse, 'noise': np.sum(noise_mask)}
        self.history['layer_4'].append(NS_data)
        return NS_data
    
    def _L5_FX(self, NS_output):
        """L5: FX (∇U, FFT(U), flux_mag, k-sp)"""
        field = NS_output['mF']
        gx, gy, gz = np.gradient(field)
        flux_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        spatial_freq = np.abs(fftn(field))
        
        FX_data = {'flux_mag': flux_magnitude, 'mean_flux': np.mean(flux_magnitude), 'k-sp': spatial_freq}
        self.history['layer_5'].append(FX_data)
        return FX_data
    
    def _L6_CW(self, NS_output, FX_output):
        """L6: CW (mask, coord 4-8, F<0.2, QP>0.5, E-int dU/dt)"""
        field_analyzed, flux, QP = NS_output['mF'], FX_output['flux_mag'], NS_output['QP']
        
        # Voorwaarden (op basis van veldanalyse en het ECHTE veld)
        coordination = sum(np.roll(np.roll(np.roll(field_analyzed > 0.0, i, 0), j, 1), k, 2)
                           for i, j, k in itertools.product((-1, 0, 1), repeat=3) if (i, j, k) != (0, 0, 0))
        resonance_condition = abs(QP) > 0.5
        flux_condition = flux > np.percentile(flux, 70)
        coordination_condition = (coordination >= 4) & (coordination <= 8)
        field_condition = self.CE_field < 0.2
        creation_mask = coordination_condition & field_condition & resonance_condition & flux_condition
        
        # E-int dU/dt (Euler-integratie)
        field_delta = np.zeros_like(self.CE_field)
        update_magnitude = (0.3 * PHI * abs(QP) * flux) * self.dt
        field_delta[creation_mask] = update_magnitude[creation_mask] * (1.0 - self.CE_field[creation_mask])
        decay_rate = 0.005 
        field_delta[~creation_mask] -= decay_rate * self.CE_field[~creation_mask] * self.dt
        
        self.CE_field += field_delta
        self.CE_field = np.clip(self.CE_field, -1.0, 1.0 * PHI)
        
        CW_data = {'mask': creation_mask, 'new_s': int(np.sum(creation_mask)), 'delta_u_dt': np.mean(np.abs(field_delta) / self.dt)}
        self.history['layer_6'].append(CW_data)
        return CW_data

    def _L7_C9(self, NS_output, FX_output, CW_output, time_s):
        """L7: C9 (Σ9, spec_coh, P-spec, α-scaling)"""
        field, flux, new_s, QP = self.CE_field, FX_output['mean_flux'], CW_output['new_s'], NS_output['QP']
        
        field_fft = fftn(field)
        power_spectrum = np.abs(field_fft)**2
        spectral_coherence = np.std(power_spectrum) / (np.mean(power_spectrum) + 1e-10) # spec_coh
        
        # Σ9: Order parameter
        order_parameter = (
            (1.0 / (spectral_coherence + 1.0)) * # Hogere orde = lagere std dev
            (1 + flux**2) *
            (1 + 0.1 * new_s) *
            PHI**abs(np.sin(time_s / ALPHA_INV)) * # α-scaling
            (1 + abs(QP)) * 1000 
        )
        
        C9_data = {'Σ9': order_parameter, 'spec_coh': spectral_coherence, 'total_s': sum(x['new_s'] for x in self.history['layer_6'])}
        self.history['layer_7'].append(C9_data)
        return C9_data
    
    # ========== LAGEN 8 & 9 (Documentatie & Output) ==========

    def _L8_DX(self, CE, GE, PH, NS, FX, CW, C9, time_s):
        """
        🔷 LAAG 8: DX (Diagnostic) - Complete Systeembeschrijving
        Consolideert alle beschrijvingen en frequenties.
        """
        DX_data = {
            'timestamp': time_s,
            'layer_descriptions': {
                'CE (L1)': 'Core Engine: Bepaalt de tijdsbasis en interne klok (1Hz, 3-6-9Hz subharmonischen).',
                'GE (L2)': 'Geometry: Analyseert de ruimtelijke buurstructuur met 2x2x2 en 3x3x3 inverse kwadraat kernels.',
                'PH (L3)': 'Phi Harmonics: Past de Gulden Snede (PHI) modulatie toe op de bureninteracties.',
                'NS (L4)': 'Noise & Resonance: Injecteert externe resonantie (7.83Hz Schumann) en stochastische ruis (1/137) en geomagnetische pulsaties.',
                'FX (L5)': 'Flux: Analyseert de energiedichtheid en ruimtelijke ordening via gradiënten en FFT spectrum (k-ruimte).',
                'CW (L6)': 'Creation Window: Drijft de structuurvorming en veldevolutie aan (Euler-integratie dU/dt) op basis van resonantie- en flux-criteria.',
                'C9 (L7)': 'Coherence: Berekent de globale orde parameter (Σ9) en systeemcoherentie (spectrale coherentie).'
            },
            'frequency_assignments': {
                'CE (L1)': ['1Hz basisritme', '3Hz', '6Hz', '9Hz subharmonischen'],
                'GE (L2)': ['structurele resonanties 2x2x2 en 3x3x3'],
                'PH (L3)': ['PHI harmonische verhouding'],
                'NS (L4)': ['7.83Hz Schumann', '3-6-9Hz quantum', '1/137 stochastic', '0.1-0.2Hz geomagnetic'],
                'FX (L5)': ['k-ruimte spatiale frequenties', 'gradiënt spectrum'],
                'CW (L6)': ['Combinatie temporele/ruimtelijke frequenties', 'Euler-integratie dU/dt'],
                'C9 (L7)': ['spectrale coherentie bands', f'alpha_inv ({ALPHA_INV:.3f}) tijdschaling']
            },
            'current_state': {
                'time': time_s,
                'Σ9': C9['Σ9'],
                'actieve_structuren': CW['new_s'],
                'mean_flux': FX['mean_flux'],
                'spectral_coherence': C9['spec_coh']
            }
        }
        self.history['layer_8'].append(DX_data)
        return DX_data

    def _L9_UN(self, C9_output, DX_output, time_s):
        """🔷 LAAG 9: UN (Universe Output) - Fysieke Manifestatie"""
        UN_data = {
            'observables': {
                'veld_snapshot': self.CE_field.copy(),
                'Σ9_timeseries': [x['Σ9'] for x in self.history['layer_7']],
                'structuur_count_timeseries': [x['new_s'] for x in self.history['layer_6']],
            },
            'physical_interpretation': {
                'totale_structuren': C9_output['total_s'],
                'systeem_coherentie': C9_output['spec_coh'],
            }
        }
        self.history['layer_9'].append(UN_data)
        return UN_data

# ========== RUN ANALYSE ==========

def run_complete_analysis(steps=500):
    """Voert de analyse uit en print de status"""
    print("🚀 QCS: 9-Lagen Quantum Coherentie Systeem - Analyse Gestart\n")
    universe = NineLayerQuantumSystem(N=137, temperature=300, delta_t=0.01)
    
    print("Stap | Orde Σ9 | Structuren | dU/dt | Coherentie")
    print("-" * 60)
    
    for step in range(steps):
        output = universe.execute_full_cycle()
        
        if step % 50 == 0:
            current = universe.history['layer_8'][-1]['current_state']
            delta_u = universe.history['layer_6'][-1]['delta_u_dt']
            
            print(f"{step:4d} | {current['Σ9']:8.1f} | {current['actieve_structuren']:10d} | {delta_u:5.3f} | {current['spectral_coherence']:6.4f}")
    
    # Finale status en L8 frequentie-overzicht
    final_dx = universe.history['layer_8'][-1]
    
    print(f"\n✅ Analyse voltooid! Finale Orde: {final_dx['current_state']['Σ9']:.1f}")
    
    print(f"\n📊 FREQUENTIE-OVERZICHT (Laag 8):")
    print(f"{'-' * 40}")
    for layer, freqs in final_dx['frequency_assignments'].items():
        print(f"  {layer}: {', '.join(freqs)}")
    return universe

if __name__ == "__main__":
    run_complete_analysis(steps=500)
