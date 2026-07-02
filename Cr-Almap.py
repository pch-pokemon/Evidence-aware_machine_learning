import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ======================
# 全局绘图风格
# ======================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 28
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300

# ======================
# 读取数据
# ======================
df = pd.read_csv("mapping_data.csv")

# ======================
# temperature 分箱
# ======================
bins = [0, 850, 950, 1050, 2000]
labels = ["<850", "850-950", "950-1050", ">1050"]
df["temp_bin"] = pd.cut(df["Temperature"], bins=bins, labels=labels)

# ======================
# quadrant 划分
# ======================
med_threshold = 0.435
ped_threshold = 0.385

def assign_quadrant(row):
    if row["P_distance"] <= ped_threshold and row["MED"] <= med_threshold:
        return "Q1"
    elif row["P_distance"] > ped_threshold and row["MED"] <= med_threshold:
        return "Q2"
    elif row["P_distance"] <= ped_threshold and row["MED"] > med_threshold:
        return "Q3"
    else:
        return "Q4"

df["quadrant"] = df.apply(assign_quadrant, axis=1)

# ======================
# 点大小映射
# ======================
min_size = 50
max_size = 220

err_min = df["abs_error"].min()
err_max = df["abs_error"].max()

if err_max == err_min:
    df["marker_size"] = (min_size + max_size) / 2
else:
    df["marker_size"] = min_size + (df["abs_error"] - err_min) / (err_max - err_min) * (max_size - min_size)

# ======================
# 配色与 marker
# ======================
color_map = {
    "Q1": "#2E8B57",
    "Q2": "#E69F00",
    "Q3": "#0072B2",
    "Q4": "#CB0505"
}

marker_map = {
    "<850": "o",
    "850-950": "s",
    "950-1050": "^",
    ">1050": "D"
}

# ======================
# 坐标范围：自动留白
# ======================
x_margin = 1
y_margin = 1

x_min = df["x_Cr"].min() - x_margin
x_max = df["x_Cr"].max() + x_margin
y_min = df["x_Al"].min() - y_margin
y_max = df["x_Al"].max() + y_margin

# ======================
# 建图
# ======================
fig, ax = plt.subplots(figsize=(8, 6.5))

# ======================
# 构造 RGBA 渐变背景
# 下红(0.9) -> 中白(0) -> 上蓝(0.7)
# ======================
n = 800
bg = np.ones((n, 2, 4))  # RGBA，先全部设为白色

bottom_rgb = np.array([253, 193, 193]) / 255.0   # #FDC1C1
top_rgb    = np.array([183, 228, 255]) / 255.0   # #B7E4FF
white_rgb  = np.array([1.0, 1.0, 1.0])

for i in range(n):
    pos = i / (n - 1)   # 0=bottom, 1=top


    if pos <= 0.5:
        # 下半部分：红 -> 白
        t = pos / 0.5
        rgb = bottom_rgb * (1 - t) + white_rgb * t
        alpha = 0.1 * (1 - t)   # 0.1 -> 0
    else:
        # 上半部分：白 -> 蓝
        t = (pos - 0.5) / 0.5
        rgb = white_rgb * (1 - t) + top_rgb * t
        alpha = 0.3 * t         # 0 -> 0.3


    bg[i, :, 0:3] = rgb
    bg[i, :, 3] = alpha


# 画背景
ax.imshow(
    bg,
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    aspect="auto",
    zorder=0
)

# ======================
# 画散点
# ======================
for q in ["Q1", "Q2", "Q3", "Q4"]:
    for t in labels:
        subset = df[(df["quadrant"] == q) & (df["temp_bin"] == t)]

        if len(subset) == 0:
            continue

        ax.scatter(
            subset["x_Cr"],
            subset["x_Al"],
            color=color_map[q],
            marker=marker_map[t],
            s=subset["marker_size"],
            alpha=0.92,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

# ======================
# 坐标轴范围
# ======================
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# ======================
# 标签与标题
# ======================
ax.set_xlabel("Cr (at.%)", fontsize=24, fontweight="bold")
ax.set_ylabel("Al (at.%)", fontsize=24, fontweight="bold")
# ax.set_title("Reliability map in Cr–Al composition space", fontsize=20, fontweight="bold")

# 坐标刻度
ax.tick_params(axis="both", labelsize=22, width=1.2)

# 淡网格
ax.grid(alpha=0.5, linestyle="--", linewidth=0.5)


#########################
# 淡网格
ax.grid(alpha=0.12, linestyle="--", linewidth=0.6)

## ======================
## 添加 legend
## ======================
#from matplotlib.lines import Line2D
#
## ---------
## 1 Temperature legend
## ---------
#temp_handles = [
#    Line2D([0],[0],
#           marker=marker_map[t],
#           color='w',
#           markerfacecolor="gray",
#           markeredgecolor="black",
#           markersize=12,
#           linestyle="None",
#           label=t)
#    for t in labels
#]
#
#legend_temp = ax.legend(
#    handles=temp_handles,
#    title="Temperature (°C)",
#    loc="upper right",
#    frameon=True,
#    fontsize=16
#)
#
#ax.add_artist(legend_temp)
#
#
## ---------
## 2 abs_error size legend
## ---------
#size_levels = np.linspace(err_min, err_max, 3)
#
#size_handles = [
#    plt.scatter([],[],
#                s=min_size + (v-err_min)/(err_max-err_min)*(max_size-min_size),
#                color="lightgray",
#                edgecolor="black",
#                label=f"{v:.2f}")
#    for v in size_levels
#]
#
#legend_size = ax.legend(
#    handles=size_handles,
#    title="|Δlgkp|",
#    loc="lower right",
#    frameon=True,
#    fontsize=16
#)
#
#ax.add_artist(legend_size)
#
#plt.tight_layout()
#########################

plt.tight_layout()

# ======================
# 保存
# ======================
plt.savefig("reliability_map.png", dpi=600, bbox_inches="tight")


plt.show()