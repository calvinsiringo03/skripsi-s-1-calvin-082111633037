import pandas as pd
import matplotlib.pyplot as plt
import io

# Data CSV InfoNCE Loss
csv_data = """epoch	global_step	loss	lr	amp_scale	enc_q_l2_norm	ema_q_k_l2_dist	grn_gamma_abs_mean	queue_ptr	queue_feat_std	queue_effective_rank	queue_neg_sim_mean	queue_neg_sim_std	file_size_mb
5	62815	7.28319	0.0001991523614	16384	225.0596	14.600583	0	44800	0.086618	40.57	0.032054	0.1856	485.62
10	125632	6.943417	0.0001897892894	32768	321.864	14.879802	0	24064	0.08826	64.88	0.00037	0.1438	485.62
15	188448	6.758042	0.0001710432806	32768	390.2166	13.802154	0	3328	0.088285	82.47	-0.000163	0.1231	485.62
20	251265	6.658086	0.0001449888024	65536	437.9613	11.898755	0	48128	0.088316	95.5	0.000104	0.1122	485.62
25	314081	6.597002	0.0001145090907	65536	469.4866	9.683717	0	27392	0.088332	105.55	0.000173	0.105	485.62
30	376897	6.555298	8.30E-05	32768	487.9681	7.251263	0	6656	0.088344	112.38	1.10E-05	0.1005	485.62
35	439714	6.526352	5.39E-05	32768	497.4639	4.718149	0	51456	0.088359	116.72	-8.30E-05	0.0976	485.62
40	502530	6.505743	3.04E-05	32768	501.4962	2.761057	0	30720	0.088363	119.39	3.70E-05	0.0958	485.62
45	565344	6.491107	1.53E-05	32768	502.9763	1.440085	0	9984	0.088365	120.9	0.00012	0.0947	485.62
50	628159	6.485497	1.00E-05	65536	503.516	0.960145	0	54784	0.088368	121.76	8.60E-05	0.0944	485.62"""

# Membaca data CSV menggunakan pandas
df = pd.read_csv(io.StringIO(csv_data), sep='\t')

# Mengatur figure matplotlib
plt.figure(figsize=(10, 6), dpi=120)

# Membuat plot garis dengan marker
plt.plot(df['epoch'], df['loss'], 
         marker='o', linestyle='-', color='#1f77b4', 
         linewidth=2.5, markersize=8, label='InfoNCE Loss')

# Menambahkan judul dan label
plt.title("Kurva InfoNCE Loss terhadap Epoch", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("InfoNCE Loss", fontsize=12)

# Mengatur grid
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(df['epoch'])

# Memberikan anotasi pada titik loss awal (Epoch 5) dan akhir (Epoch 50)
plt.annotate(f"{df['loss'].iloc[0]:.4f}", 
             (df['epoch'].iloc[0], df['loss'].iloc[0]), 
             textcoords="offset points", xytext=(10, 5), ha='left', 
             fontsize=10, fontweight='bold', color='#333333')

plt.annotate(f"{df['loss'].iloc[-1]:.4f}", 
             (df['epoch'].iloc[-1], df['loss'].iloc[-1]), 
             textcoords="offset points", xytext=(10, 5), ha='left', 
             fontsize=10, fontweight='bold', color='#333333')

# Menghilangkan garis kotak (spines) hitam di sekeliling grafik
ax = plt.gca()
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)

# Merapikan layout
plt.tight_layout()

# Menampilkan grafik (Siap di-run di VS Code)
plt.show()