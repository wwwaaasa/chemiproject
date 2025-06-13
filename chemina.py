# ===== 라이브러리 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
plt.rcParams['axes.unicode_minus'] = False

# ===== 데이터 불러오기 =====
csv_path = r"C:\Users\samsung\OneDrive\문서\수행평가\화학 수행\3학년\dataset\energy1(Na1).txt"

colnames = ['Configuration', 'Term', 'J', 'Prefix', 'Level (eV)', 'Suffix', 'Uncertainty (eV)', 'Leading percentages', 'Reference']

df = pd.read_csv(
    csv_path,
    sep=',',
    engine='python',
    header=None,
    names=colnames,
    skipinitialspace=True,
    skiprows=1  # 첫 줄 header 깨짐 방지
)

print("컬럼 확인:", df.columns.tolist())

# ===== 데이터 전처리 =====
for col in ['Configuration', 'Term', 'J', 'Level (eV)']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip('="')

# E_exp 컬럼 준비
df['E_exp'] = df['Level (eV)'].str.extract(r'([\d\.\-E]+)').astype(float)

# J 값 변환 (fraction 처리)
df['J'] = df['J'].str.extract(r'([\d/\.]+)')
df['J'] = df['J'].apply(lambda x: eval(x) if pd.notnull(x) else np.nan)

# n 값 추출
df['n'] = df['Configuration'].str.extract(r'(\d+)')
df['n'] = df['n'].astype(float)
df['n'] = df['n'].fillna(0).astype(int)

# ℓ 값 추출
df['l_char'] = df['Configuration'].str.extract(r'\d*([spdfg])')
l_map = {'s':0,'p':1,'d':2,'f':3,'g':4}
df['ℓ'] = df['l_char'].map(l_map)

# ===== Rydberg 준위 선택 (n ≥ 2) =====
ry = df[df['n'] >= 2].copy()

# ===== 이론 에너지 계산 =====

# Constants
Rhc  = 13.605693
m_e  = 9.10938356e-31
c    = 2.99792458e8
hbar = 1.054571817e-34
e    = 1.602176634e-19
eps0 = 8.8541878128e-12
alpha= e**2/(4*np.pi*eps0*hbar*c)

# Schrödinger 계산
ry['E_schr'] = -Rhc / ry['n']**2

def E_dirac(Z, n, j):
    a = Z * alpha
    kappa = j + 0.5
    gamma_squared = kappa**2 - a**2
    if gamma_squared < 0:
        return np.nan
    gamma = np.sqrt(gamma_squared)
    denom = n - kappa + gamma
    E_rel = m_e * c**2 * (1 + (a/denom)**2)**(-0.5) - m_e * c**2
    return E_rel / e


Z_Na = 11
ry['E_dirac'] = ry.apply(lambda r: E_dirac(Z_Na, r['n'], r['J']), axis=1)


# ===== ΔE 계산 =====

# ΔE_exp 계산
ry['DeltaE_exp'] = ry['E_exp'].max() - ry['E_exp']
ry['DeltaE_schr'] = ry['E_schr'].max() - ry['E_schr']
ry['DeltaE_dirac'] = ry['E_dirac'].max() - ry['E_dirac']

# ===== NaN 제거 =====
ry_valid = ry.dropna(subset=['DeltaE_exp', 'DeltaE_schr', 'DeltaE_dirac'])

# ===== 선형 회귀 =====

X_exp = ry_valid['DeltaE_exp'].values.reshape(-1,1)

# Schrödinger ΔE 학습
lr_schr = LinearRegression()
lr_schr.fit(X_exp, ry_valid['DeltaE_schr'].values.reshape(-1,1))
slope_schr = lr_schr.coef_[0][0]

# Dirac ΔE 학습
lr_dirac = LinearRegression()
lr_dirac.fit(X_exp, ry_valid['DeltaE_dirac'].values.reshape(-1,1))
slope_dirac = lr_dirac.coef_[0][0]



# ===== 첫 번째 그래프 (raw E 그래프, 점 + 선 모두 Shifted 적용 + 실험값 점 표시) =====
plt.figure(figsize=(8,8))

C_shift_raw = -5  # 원하는 만큼 shift 적용

# 실험 vs 슈뢰딩거 → 점도 shift 적용
plt.scatter(ry_valid['E_exp'], ry_valid['E_schr'] + C_shift_raw, label='Schrödinger (raw shifted)', alpha=0.8, color='blue', s=100)

# 실험 vs 디랙 → 점도 shift 적용
plt.scatter(ry_valid['E_exp'], ry_valid['E_dirac'] + C_shift_raw, label='Dirac (raw shifted)', alpha=0.6, color='orange', s=100)

# 실험값 점 (optional, 원하면 활성화)
plt.scatter(ry_valid['E_exp'], ry_valid['E_exp'] + C_shift_raw, label='실험값 (Shifted)', alpha=0.5, color='green', s=70)

# Perfect Fit shifted
minv_raw = ry_valid['E_exp'].min()
maxv_raw = ry_valid['E_exp'].max()
plt.plot([minv_raw, maxv_raw],[minv_raw + C_shift_raw,maxv_raw + C_shift_raw],'k--',label='완벽 일치선 (Shifted, 기울기=1)')

# 스케일 자동/적절 조정 → 너무 넓게 잡으면 이상하게 보임 → 약간 타이트하게
plt.xlim(minv_raw - 0.5, maxv_raw + 0.5)
plt.ylim((ry_valid[['E_schr','E_dirac','E_exp']].min().min() + C_shift_raw) - 1,
         (ry_valid[['E_schr','E_dirac','E_exp']].max().max() + C_shift_raw) + 1)

# 축, 제목
plt.xlabel("실험 E_exp (eV)")
plt.ylabel("이론 E (eV) + Shift")

plt.title("H I: Schrödinger vs Dirac vs 실험 (raw E, 점+선 모두 Shifted 적용 + 실험값 표시)")

# 범례
plt.legend()

# 출력
plt.tight_layout()
plt.show()




# ===== ΔE 그래프 (Perfect Fit shifting 포함) =====
plt.figure(figsize=(8,8))

# Schrödinger ΔE 점 + 회귀선
plt.scatter(ry_valid['DeltaE_exp'], ry_valid['DeltaE_schr'], label=f'Schrödinger ΔE (slope={slope_schr:.4f})', alpha=0.6, color='blue', s=50)
plt.plot(ry_valid['DeltaE_exp'], lr_schr.predict(X_exp), color='blue', linestyle='--')

# Dirac ΔE 점 + 회귀선
plt.scatter(ry_valid['DeltaE_exp'], ry_valid['DeltaE_dirac'], label=f'Dirac ΔE (slope={slope_dirac:.4f})', alpha=0.6, color='orange', s=50)
plt.plot(ry_valid['DeltaE_exp'], lr_dirac.predict(X_exp), color='orange', linestyle='--')

# Perfect Fit shifted
C_shift = -0.2  # ΔE 그래프에서 shift 적용
minv = ry_valid['DeltaE_exp'].min()
maxv = ry_valid['DeltaE_exp'].max()
plt.plot([minv, maxv], [minv + C_shift, maxv + C_shift], 'k--', label='완벽 일치선 (Shifted, 기울기=1)')

# 스케일 자동
plt.xlim(minv, maxv)
plt.ylim(minv + C_shift - 0.1, maxv + C_shift + 0.1)

# 한글 축
plt.xlabel("실험 ΔE (eV)")
plt.ylabel("이론 ΔE (eV)")

# 제목
plt.title("H I ΔE 비교: 슈뢰딩거 vs 디랙 vs 실험 (Perfect Fit shifting 포함)")

# 범례
plt.legend()

# 출력
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 절대 에너지 E_n vs n 그래프 (수정 최종판) — 오류 방지!
# ------------------------------------------------------

# 우선 ry 전체에서 NaN 제거하고 사용
df_plot_abs = ry[['n', 'E_exp', 'E_schr', 'E_dirac']].dropna()

# 그룹핑: n 별 평균 절대 에너지 계산
df_grouped_abs = df_plot_abs.groupby('n').agg({
    'E_exp': 'mean',   # ← 수정됨
    'E_schr': 'mean',
    'E_dirac': 'mean'
}).reset_index()

# 그래프 그리기
plt.figure(figsize=(8,6))

plt.plot(df_grouped_abs['n'], df_grouped_abs['E_exp'], 'ko-', label='실험 E_exp (절대)', markersize=8)
plt.plot(df_grouped_abs['n'], df_grouped_abs['E_schr'], 'bo-', label='Schrödinger E (절대)', markersize=8)
plt.plot(df_grouped_abs['n'], df_grouped_abs['E_dirac'], 'ro-', label='Dirac E (절대)', markersize=8)

plt.xlabel('주양자수 n')
plt.ylabel('절대 에너지 E (eV)')
plt.title('H I: 절대 에너지 E_n vs n 비교 (실험 vs 슈뢰딩거 vs 디랙)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Z 값 설정 (Na 용)
Z_Na = 11  # Na 원자

# Dirac 에너지 함수 (Z 값 넣도록 수정)
def E_dirac_general(Z, n, j):
    a = Z * alpha
    kappa = j + 0.5
    gamma = np.sqrt(np.maximum(0, kappa**2 - a**2))
    denom = n - kappa + gamma
    E_rel = m_e * c**2 * (1 + (a/denom)**2)**(-0.5) - m_e * c**2
    return E_rel / e

def E_schr(Z, n):
    return -Rhc * Z**2 / n**2

ry['E_schr'] = ry['n'].apply(lambda n: E_schr(Z_Na, n))

# 원하는 n 값 범위
n_values = np.arange(2, 13)  # n=2~12

# 계산 결과 저장용 리스트
E_dirac_Na = []
E_schr_Na  = []

for n in n_values:
    E_dirac_Na.append(E_dirac_general(Z_Na, n, 0.5))  # 예시로 j=0.5
    E_schr_Na.append(E_schr_general(Z_Na, n))


delta_E = np.abs(np.array(E_schr_Na)) - np.abs(np.array(E_dirac_Na))

plt.figure(figsize=(8,5))
plt.plot(n_values, delta_E, 'mo-', label="ΔE = |Schrödinger| - |Dirac|")
plt.xlabel("주양자수 n")
plt.ylabel("ΔE (eV)")
plt.title("Na I: ΔE vs n (Schrödinger - Dirac)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()