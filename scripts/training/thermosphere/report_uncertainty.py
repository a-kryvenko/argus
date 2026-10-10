"""Report actual held-out coverage of upper ap scenarios and density bounds."""
import html
import json
from pathlib import Path

from report import plt
import pandas as pd

STYLES = [("rho_q50", "ap q50", "#7356b3"), ("rho_q90", "ap q90 scenario", "#d06632"),
          ("rho_q95", "ap q95 scenario", "#b44368"), ("density_upper90", "Calibrated density U90", "#147a88"),
          ("density_upper95", "Calibrated density U95", "#477b34")]


def write_report(directory):
    directory = Path(directory)
    info = json.loads((directory/"results.json").read_text())
    bounds = pd.read_csv(directory/"bound_metrics.csv")
    intervals = pd.read_csv(directory/"interval_metrics.csv")
    parameters = pd.read_csv(directory/"density_calibration.csv")
    coverage = pd.read_csv(directory/"test_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, label, color in STYLES:
        for ax, grouping, activity in [(axes[0], "lead_hours", None), (axes[1], "lead_hours+activity", "storm")]:
            selection = bounds.loc[bounds.grouping.eq(grouping) & bounds.scenario.eq(name)]
            if activity:
                selection = selection.loc[selection.activity.eq(activity)]
            ax.plot(selection.lead_hours, selection.coverage_pct, "o-", color=color, label=label)
    for ax, title in zip(axes, ["All samples", "Storm samples: verification Kp ≥ 5"]):
        ax.axhline(90, color="grey", linestyle="--", linewidth=.8)
        ax.axhline(95, color="grey", linestyle=":", linewidth=.8)
        ax.set(title=title, xlabel="Lead (hours)", ylabel="Truth below scenario / upper bound (%)", ylim=(0, 101))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(directory/"coverage.png", dpi=180)
    plt.close(fig)
    h24 = bounds.loc[bounds.lead_hours.eq(24)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    storm = h24.loc[h24.activity.eq("storm")].set_index("scenario")
    for name, label, color in STYLES:
        row = storm.loc[name]
        axes[0].bar(label, row.underestimate_gt_30_pct, color=color)
        axes[1].bar(label, row.overestimate_gt_50_pct, color=color)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=.2)
    axes[0].set(title="Storms, 24 h: underestimation >30%", ylabel="Samples (%)")
    axes[1].set(title="Storms, 24 h: overestimation >50%", ylabel="Samples (%)")
    fig.savefig(directory/"storm_tradeoff.png", dpi=180)
    plt.close(fig)
    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x:f"{x:.2f}")
    cols = ["activity", "scenario", "n", "coverage_pct", "underestimate_gt_30_pct", "overestimate_gt_50_pct", "mape_pct"]
    short = h24.copy();short["activity"] = short.activity.fillna("all")
    limits = "".join(f"<li>{html.escape(s)}</li>" for s in info["limitations"])
    upper = storm.loc["density_upper95"]
    content = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Верхние сценарии ap и неопределённость плотности</title>
<style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#202b36;background:#fafbfc}}img{{width:100%}}
table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}</style>
<h1>Верхние сценарии ap и калибровка неопределённости плотности</h1>
<p>q50 сохранён из предыдущего эксперимента. Дополнительно обучены ap q90/q95 на тех же данных до декабря 2024 года,
без подбора на тесте. Прогнозы с пересечением квантилей упорядочены повышением верхних сценариев; медиана сохранена.
Через DTM вычислены отдельные траектории плотности. <b>Квантиль ap не является квантилем плотности.</b></p>
<p>Отдельная калибровка: 3–28 декабря 2024, GRACE-FO 1. Для каждого горизонта рассчитаны 5-й, 90-й и 95-й
эмпирические процентили log(измеренная плотность / прогноз q50). Их экспоненты задают множители нижней границы,
U90 и U95. Центральный интервал 90%: [L05, U95]. Калибровка заморожена до расчёта тестовых результатов.</p>
<p><b>Калибровка содержит в основном спокойные условия:</b> {html.escape(str(info['calibration_activity_counts']))};
максимальный Kp {info['calibration_max_kp']:.1f}. На бурях 2025 года верхняя граница U95 на 24 часах
покрыла {upper.coverage_pct:.1f}% наблюдений, а недооценки более чем на 30% остались в {upper.underestimate_gt_30_pct:.1f}% случаев.
Номинальное название U95 не означает достигнутые 95% покрытия в новом режиме.</p>
<img src="coverage.png" alt="Покрытие на всей выборке и на бурях">
<h2>Горизонт 24 часа</h2>{table(short[cols])}
<p>Coverage — доля измерений, не превышающих сценарий/границу. Это не вероятность отсутствия риска спутнику.
Недооценка &gt;30% — значение ниже 0,7 от измерения; переоценка &gt;50% — выше 1,5 от измерения.
Последний показатель оценивает цену консерватизма, а не частоту ложных операционных тревог.</p>
<img src="storm_tradeoff.png" alt="Недооценка и переоценка на бурях">
<h2>Калибровочные множители</h2>{table(parameters)}
<h2>Центральный интервал 90% на тесте</h2>{table(intervals)}
<p>Интервальный score учитывает ширину и промахи границ; меньше — лучше. Покрытие эмпирическое,
без гарантии для зависимых временных рядов или экстремальных бурь.</p>
<h2>Все верхние границы по горизонтам</h2>{table(bounds)}
<h2>Покрытие данных</h2>{table(coverage)}
<p>Проверено численное совпадение q50 с предыдущим экспериментом. Калибровочный период заканчивается
{html.escape(info['calibration_max_valid_time'])}, раньше первого тестового выпуска {html.escape(info['test_min_issue'])}.</p>
<h2>Ограничения</h2><ul>{limits}</ul>
<h2>Повторение</h2><pre>.venv/bin/python scripts/training/thermosphere/prepare_uncertainty.py --download --train
.venv/bin/python -u scripts/training/thermosphere/run_uncertainty.py
.venv/bin/python -m pytest scripts/training/thermosphere -q</pre>
<p>Параметры — config_uncertainty.json; контрольные суммы — results.json и training.json;
calibration_predictions.parquet и test_predictions.parquet содержат раздельные результаты.
Рабочая цепочка не изменена.</p></html>"""
    (directory/"report.html").write_text(content)


if __name__ == "__main__":
    from prepare_uncertainty import OUT
    write_report(OUT)
