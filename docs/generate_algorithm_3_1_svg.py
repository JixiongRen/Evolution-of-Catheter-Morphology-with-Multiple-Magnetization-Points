from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parent
SVG = OUT / "algorithm_3_1_fk.svg"

W = 1800
M = 32
TITLE_SIZE = 40
BODY_SIZE = 31
FORMULA_SIZE = 30
SMALL_SIZE = 28
CN_FONT = "'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'DejaVu Sans', sans-serif"
MATH_FONT = "'Liberation Serif', 'DejaVu Serif', 'Times New Roman', serif"


def text(x: int, y: int, s: str, size: int = BODY_SIZE, weight: str = "400", style: str = "normal",
         family: str = CN_FONT) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" font-style="{style}" fill="#000">{escape(s)}</text>'
    )


def line(y: int, stroke_width: float) -> str:
    return f'<line x1="{M}" y1="{y}" x2="{W-M}" y2="{y}" stroke="#000" stroke-width="{stroke_width}"/>'


def main() -> None:
    parts: list[str] = []
    y = 52
    parts.append(line(18, 4))
    parts.append(text(M + 6, y, "算法3-1  多磁化点导管正运动学 FK-LM 求解", TITLE_SIZE, "700"))
    y += 55

    label_x = M + 6
    content_x = M + 190
    parts.append(text(label_x, y, "输入：", BODY_SIZE, "700"))
    input_lines = [
        "柔性段参数， F={L_i, M_i, K_se,i, K_bt,i, v_i*, u_i*}_{i=1}^N",
        "刚性段参数， R={L_r,i, m_i^b}_{i=1}^N",
        "近端边界， x_b0=(p_0, Q_0)；线圈电流， I；重力， g",
        "收敛容差， epsilon；最大迭代步数， K_max",
    ]
    for k, s in enumerate(input_lines):
        parts.append(text(content_x, y + 43 * k, s, BODY_SIZE))
    y += 43 * len(input_lines)

    parts.append(text(label_x, y, "输出：", BODY_SIZE, "700"))
    output_lines = [
        "平衡状态， z*",
        "中心线， C(s)；磁体位姿， {(p_m,i, R_m,i)}_{i=1}^N",
    ]
    for k, s in enumerate(output_lines):
        parts.append(text(content_x, y + 43 * k, s, BODY_SIZE))
    y += 70
    parts.append(line(y, 2.2))
    y += 48

    step_x = M + 80
    body_x = M + 165
    formula_x = M + 470

    def step(n: int, title: str, formulas: list[str], gap_after: int = 30) -> None:
        nonlocal y
        parts.append(text(step_x, y, f"{n}.", BODY_SIZE))
        parts.append(text(body_x, y, f"{title}：", BODY_SIZE, "700"))
        yy = y + 5
        if len(formulas) == 1:
            parts.append(text(formula_x, yy, formulas[0], FORMULA_SIZE, "600", "italic", MATH_FONT))
            y += 58 + gap_after
            return
        for f in formulas:
            parts.append(text(formula_x, yy, f, FORMULA_SIZE, "600", "italic", MATH_FONT))
            yy += 41
        y = yy + gap_after

    step(1, "初始化", [
        "F_bar, R_bar, g_bar <- Nondim(F, R, g)",
        "z_0={x_f,i,j, x_r,i, eta_i},   lambda_0>0,   k=0",
    ], 24)
    step(2, "状态解包", [
        "{x_f,i,j}, {x_r,i}, {eta_i} <- Unpack(z_k)",
    ], 24)
    step(3, "柔性段", [
        "x_hat_f,i,j+1 = Phi_f(x_f,i,j; K_se,i, K_bt,i, v_i*, u_i*, g_bar)",
        "E_f,i,j = x_f,i,j+1 - x_hat_f,i,j+1",
    ], 24)
    step(4, "刚性段", [
        "m_i = R_r,i m_i^b",
        "(f_m,i, tau_m,i) = Psi_m(p_m,i, R_r,i, m_i, I)",
        "E_r,i = x_r,i+ - Phi_r(x_r,i-; f_m,i, tau_m,i, g_bar, L_r,i)",
    ], 24)
    step(5, "耦合", [
        "E_joint,i = [p_f,i^e-p_r,i-, Log((Q_r,i-)^{-1}Q_f,i^e),",
        "             n_f,i^e-n_r,i-, tau_f,i^e-tau_r,i-]",
    ], 24)
    step(6, "边界", [
        "E_bc = [p_f,1,0-p_0, Log(Q_0^{-1}Q_f,1,0)]",
        "E_tip = [n_N^e, tau_N^e]",
    ], 24)
    step(7, "自动微分", [
        "E_k = [E_f, E_r, E_joint, E_bc, E_tip]^T",
        "J_k = dE/dz |_(z=z_k)",
    ], 24)
    step(8, "LM步", [
        "(J_k^T J_k + lambda_k I) Delta z_k = -J_k^T E_k",
        "z_trial = z_k + Delta z_k",
    ], 24)
    step(9, "阻尼更新", [
        "if  ||E(z_trial)||_2 < ||E_k||_2:",
        "    z_k+1 = z_trial,    lambda_k+1 = beta_down lambda_k",
        "else:",
        "    z_k+1 = z_k,        lambda_k+1 = beta_up lambda_k",
    ], 24)
    step(10, "停止", [
        "if  ||E(z_k+1)||_2 < epsilon  or  k = K_max:",
        "    output  z*, C(s), {(p_m,i, R_m,i)}",
        "else:",
        "    k <- k+1, return Step 2",
    ], 0)

    h = y + 38
    parts.append(line(h - 18, 4))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
        f'viewBox="0 0 {W} {h}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    SVG.write_text(svg, encoding="utf-8")
    print(SVG)


if __name__ == "__main__":
    main()
