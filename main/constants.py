IGF = 'IGF-1 ng/ml (N: 100-311)'
VITAMINE_D = 'vitamin 25-OH D ng/ml'
PROLACTIN = 'prolactin'
PCO = 'PCO 0-healthy control, 1-PCOS, 2-FHA 3-POF, 4-High Andro'

CLASS_NAMES = ['healthy', 'PCOS', 'FHA', 'High Andro']


CATEGORICAL_ATTRIBUTES = [
    'PCO 0-healthy control, 1-PCOS, 2-FHA 3-POF, 4-High Andro', 'Hypertension', 'WC>88', 'WHR>0,85 (WHO)', 'WHR>0,8 (NIDDK)', 'WHTR>0,5', 
    'overweight/obesity 0-normal/low, 1-overweight, 2-obesity',
    'irregular cycles (0-nie, 1-tak)', 'ovulation (0-brak, 1-obecna)', 'PCO ovary morfology in USG (0-brak, 1--obecna)',
    'stromal hypertrophy in ovary (0-brak, 1-obecny)', 'acne', 'hirsutism', 'hyperandrogenism', 'elevated DHT', 'hypothyroidism', 'nodules 0-lack, 1-RL,  2-LL, 3-both', 'chronic thyroiditis',
    'follicules >12', 
    # 'hyperlipidemia', 'elevated LDL and TG', 
    'CHOL>200', 'HDL<50', 'LDL>135', 'SHBG>110',
    # 'TG>150', 'Impaired Fasting Glucose ', 'Impaired Glucose Tolerance', 
    # 'month of birth', 'quarter of the year',
    ]



CONTINUOUS_ATTRIBUTES = ['IGF-1 ng/ml (N: 100-311)',
 'proBNP',
 'AMH (ng/ml) *7,14=pmol/l',
 'weight',
 'height (cm)',
 'BMI',
 'systolic BP (ciśnienie skurczowe)',
 'diastolic BP (ciśnienie rozskurczowe)',
 'Waist Circumference (WC)',
 'Hip Circumference (HC)',
 'WHR (Waist/Hip ratio)',
 'WHTR (Waist/Height Ratio)',
 'FG score (Ferriman-Gallway score - stopień androgenizacji)',
 'Volume of the thyroid  Right Lobe',
 'Volume of the thyroid  Left Lobe',
 'thyroid volume',
 'Vole of the Right Ovary',
 'Volume of the  Left Ovary',
 'ovaries volume - total',
 'WBC x10^3/ul',
 'neutrophil x10^3/ul',
 'lymphocytes x10^3/ul',
 'monocytes x10^3/ul',
 'eosinocytes x10^3/ul',
 'basophils x10^3/ul',
 '% neutrophil ',
 '% lymphocytes ',
 '% monocytes',
 '%eosinocytes ',
 '%basophils ',
 'RBC x10^6ul',
 'Hemoglobin [g/dl]',
 'hematocrit [%]',
 'HTC/Hb',
 'MCV fl',
 'MCH pg',
 'MCHC g/dl',
 'RDW-CV %',
#  'NRBC x10^3/ul',
 'PLT x10^3/ul',
 'PDW fl',
 'MPV fl',
 'P-LCR %',
 'PLT/WBC',
 'MPV/PLT',
 'PLR',
 'limf/mono',
 'NLR (stosunek neutrofili do limfocytów)',
 'L/WCC (leukocyty do całkowitej liczby krwinek białych)',
 'eos/leukocyty',
 'sodium mmol/l',
 'potassium mmol/l',
 'calcium mg/dl',
 'phosphorus mg/dl',
 'creatinine mg/dl',
 'CRP mg/l',
 'ALT U/l',
 'AST U/l',
 'Bilirubin mg/dl',
#  'CHOL mg/dl ',
 'CHOL mmol/l',
#  ' HDL mg/dl',
 'HDL mmol/l',
#  'LDL mg/dl',
 'LDL mmol/l',
#  'TG',
 'TG mmol/l',
 'Atherogenic index (AI) (LDL-C/HDL-C) ',
 'coronary risk index (CRI) (TG/HDL-C)',
 'VAI - Visceral adiposity index',
 'BAI - Body adiposity index',
 'LAP INDEX - Lipid accumulation product index',
 'TyG Index - Trigliceride-glucose index',
 'AIP -Atherogenic index of plasma',
 'UIBC ug/dl',
 'ferrum ug/dl',
 'TIBC',
 'TSAT',
 'ferritin ng/ml',
 'glucose 0 mg/dl',
 'glucose  120 mg/dl',
 'insulin 0 uU/ml',
 'Insulin 120 uU/ml',
 'HOMA',
 'Matsuda',
 'QUICKI (N<0,357)',
 'TSH mIU/L',
 'FT3 pmol/l',
 'FT4 pmol/l',
 'Anty-TPO IU/ml',
 'Anty-TG IU/ml',
 'FSH mlU/ml',
 'LH',
 'LH/FSH',
 'prolactin',
 'DHEA-S ug/dl',
 'testosterone nmol/l',
 'T (ng/ml)',
 'T/SHBG',
 'E(pg/ml)/T(ng/ml)/',
 'Parathormone pg/ml',
 'cortisol nmol/l  8:00',
 'cortisol nmol/l 18:00',
 'Estradiol pg/ml',
 'SHBG nmol/l',
 'FTI (free testosterone index)',
 'ACTH pg/ml',
 'HbA1c %',
 'vitamin 25-OH D ng/ml',
 'Androstendione ng/ml',
 '17-OH-progesterone ng/ml',
 'Dihydrotestosterone pg/ml (N<368)',
 'Testosterone/DHT',
 'T/A (testosterone/androstendione)',
 'age']


# parametry związane z układem krążenia i ryzykiem jego chorób
circulatory_system_attributes = [
    'proBNP', 'systolic BP (ciśnienie skurczowe)', 'diastolic BP (ciśnienie rozskurczowe)', 'Hypertension', 
    # 'hyperlipidemia', 
    # 'elevated LDL and TG',
    'CHOL mg/dl ', 'CHOL mmol/l', 'CHOL>200', ' HDL mg/dl', 'HDL mmol/l', 'HDL<50', 'LDL mg/dl', 'LDL mmol/l', 'LDL>135', 'TG', 'TG mmol/l', 
    # 'TG>150',
    'Atherogenic index (AI) (LDL-C/HDL-C) ', 'coronary risk index (CRI) (TG/HDL-C)', 'VAI - Visceral adiposity index', 'BAI - Body adiposity index',
    'LAP INDEX - Lipid accumulation product index', 'TyG Index - Trigliceride-glucose index', 'AIP -Atherogenic index of plasma',
    ]
# hormony płciowe plus morfologia jajnika
sex_hormones_attributes = [
    'AMH (ng/ml) *7,14=pmol/l', 'irregular cycles (0-nie, 1-tak)', 'ovulation (0-brak, 1-obecna)', 'PCO ovary morfology in USG (0-brak, 1--obecna)',
    'stromal hypertrophy in ovary (0-brak, 1-obecny)', 'acne', 'hirsutism', 'FG score (Ferriman-Gallway score - stopień androgenizacji)', 'hyperandrogenism', 'elevated DHT',
    'Vole of the Right Ovary', 'Volume of the  Left Ovary', 'ovaries volume - total', 'follicules >12', 'FSH mlU/ml', 'LH', 'LH/FSH', 'DHEA-S ug/dl', 'testosterone nmol/l',
    'T (ng/ml)', 'T/SHBG', 'E(pg/ml)/T(ng/ml)/', 'Estradiol pg/ml', 'SHBG nmol/l', 'SHBG>110', 'FTI (free testosterone index)', 'Androstendione ng/ml', '17-OH-progesterone ng/ml',
    'Dihydrotestosterone pg/ml (N<368)', 'Testosterone/DHT', 'T/A (testosterone/androstendione)',
    ]
# gospodarka węglowodanowa (włączając oporność i wrazliwość na insulinę)
carbohydrate_metabolism_attributes = [
    'glucose 0 mg/dl', 'glucose  120 mg/dl', 'insulin 0 uU/ml', 'Insulin 120 uU/ml', 'HOMA', 'Matsuda', 'QUICKI (N<0,357)', 
    # 'Impaired Fasting Glucose ', 'Impaired Glucose Tolerance',
    ]
# parametry antropometryczne
anthropometric_attributes = [
    'weight', 'height (cm)', 'BMI', 'Waist Circumference (WC)', 'WC>88', 'Hip Circumference (HC)', 'WHR (Waist/Hip ratio)', 'WHR>0,85 (WHO)',
    'WHR>0,8 (NIDDK)', 'WHTR (Waist/Height Ratio)', 'WHTR>0,5', 'overweight/obesity 0-normal/low, 1-overweight, 2-obesity',
    ]
# parametry tarczycowe
thyroid_attributes = [
    'hypothyroidism', 'Volume of the thyroid  Right Lobe', 'Volume of the thyroid  Left Lobe', 'thyroid volume', 'nodules 0-lack, 1-RL,  2-LL, 3-both',
    'chronic thyroiditis', 'TSH mIU/L', 'FT3 pmol/l', 'FT4 pmol/l', 'Anty-TPO IU/ml', 'Anty-TG IU/ml',
    ]
# parametry zapalne
inflammatory_attributes = [
    'WBC x10^3/ul', 'neutrophil x10^3/ul', 'lymphocytes x10^3/ul', 'monocytes x10^3/ul', 'eosinocytes x10^3/ul', 'basophils x10^3/ul', '% neutrophil ', '% lymphocytes ',
    '% monocytes', '%eosinocytes ', '%basophils ', 'RBC x10^6ul', 'Hemoglobin [g/dl]', 'hematocrit [%]', 'HTC/Hb', 'MCV fl', 'MCH pg', 'MCHC g/dl', 'RDW-CV %', 
    # 'NRBC x10^3/ul',
    'PLT x10^3/ul', 'PDW fl', 'MPV fl', 'P-LCR %', 'PLT/WBC', 'MPV/PLT', 'PLR', 'limf/mono', 'NLR (stosunek neutrofili do limfocytów)',
    'L/WCC (leukocyty do całkowitej liczby krwinek białych)', 'eos/leukocyty', 'CRP mg/l',
    ]
#  gospodarka żelazowa
iron_attributes = ['UIBC ug/dl', 'ferrum ug/dl', 'TIBC', 'TSAT', 'ferritin ng/ml']
# gospodarka wapniowo-fosforanowa
calcium_attributes = ['calcium mg/dl', 'phosphorus mg/dl', 'Parathormone pg/ml']

attribute_groups = [
    circulatory_system_attributes, sex_hormones_attributes, carbohydrate_metabolism_attributes, anthropometric_attributes,
    thyroid_attributes, inflammatory_attributes, iron_attributes, calcium_attributes,
    ]


vitamin_d_associated_features = ['creatinine mg/dl', 'Parathormone pg/ml', 'phosphorus mg/dl']
igf_associated_features = ['HOMA', 'QUICKI (N<0,357)']
prolactin_associated_features = ['LH', 'cortisol nmol/l  8:00', '17-OH-progesterone ng/ml', 'T/SHBG', 'hyperandrogenism', 
                                 'PCO ovary morfology in USG (0-brak, 1--obecna)', 'LH/FSH', 'hirsutism', 'Testosterone/DHT', 'follicules >12',
                                 'Volume of the Left Ovary', 'testosterone nmol/l', 'Androstendione ng/ml']



index_features = [
    'LAP INDEX - Lipid accumulation product index',
    'VAI - Visceral adiposity index',
    'BAI - Body adiposity index',
    'TyG Index - Trigliceride-glucose index',
    'AIP -Atherogenic index of plasma',
    'Atherogenic index (AI) (LDL-C/HDL-C) ',
    'coronary risk index (CRI) (TG/HDL-C)',
    'FTI (free testosterone index)',
]



columns_for_one_hot_encoding = {'PCO 0-healthy control, 1-PCOS, 2-FHA 3-POF, 4-High Andro': 'PCO',
                             'overweight/obesity 0-normal/low, 1-overweight, 2-obesity': 'overweight',
                             'nodules 0-lack, 1-RL,  2-LL, 3-both': 'nodules',
                             'quarter of the year': 'birth_quarter',
                             'month of birth': 'birth_month',
                             }

new_column_names_map = {
    'PCO_0': '0_healthy_control',
    'PCO_1': '1_PCOS',
    'PCO_2': '2_FHA',
    'PCO_3': '3_POF',
    'PCO_4': '4_High_Andro',
    'overweight_0': '0_normal_weight',
    'overweight_1': '1_overweight',
    'overweight_2': '2_obesity',
    'nodules_0': 'no_nodules',
    'nodules_1': 'nodules_right',
    'nodules_2': 'nodules_left',
    'nodules_3': 'nodules_both_sides',
    }