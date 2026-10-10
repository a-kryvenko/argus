"""Standalone three-variant report, with paired coverage and driver diagnostics."""
import html
import json
from pathlib import Path

from report import plt
import pandas as pd

STYLES = [("jb2008", "Current JB2008", "#d06632"),
          ("dtm2020", "DTM2020 fixed drivers", "#147a88"),
          ("dtm2020_ap_forecast", "DTM2020 forecast ap", "#7356b3")]


def write_report(directory):
    directory = Path(directory)
    scores = pd.read_csv(directory/"metrics.csv")
    coverage = pd.read_csv(directory/"coverage.csv")
    differences = pd.read_csv(directory/"paired_differences.csv")
    info = json.loads((directory/"results.json").read_text())
    data = pd.read_parquet(directory/"predictions.parquet").dropna(subset=[s[0] for s in STYLES])
    overall = scores.loc[scores.grouping.eq("lead_hours")]
    storm_scores = scores.loc[scores.grouping.eq("lead_hours+activity") & scores.activity.eq("storm")]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, label, color in STYLES:
        subset = overall.loc[overall.model.eq(name)]
        axes[0].plot(subset.lead_hours, subset.mape_pct, "o-", color=color, label=label)
        storm = storm_scores.loc[storm_scores.model.eq(name)]
        axes[1].plot(storm.lead_hours, storm.underestimate_gt_30pct*100, "o-", color=color)
    for ax in axes:
        ax.set_xlabel("Forecast lead (hours)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean absolute percentage error (%)")
    axes[1].set_ylabel("Storm-time underestimation >30% (% of samples)")
    axes[1].set_title("Verification-time Kp ≥ 5")
    axes[0].legend(fontsize=9)
    fig.savefig(directory/"comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for ax, (window, group) in zip(axes, data.loc[data.lead_hours.eq(24)].groupby("window")):
        averaged = group.set_index("valid_time")[["truth", *[s[0] for s in STYLES]]].resample("6h").mean()*1e12
        for name, label, color in [("truth", "GRACE-FO", "#222222"), *STYLES]:
            ax.plot(averaged.index, averaged[name], color=color, label=label)
        ax.set(title=f"{window}; 24 h lead; 6 h display averages", ylabel="Density (10⁻¹² kg/m³)")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2, fontsize=9)
    fig.savefig(directory/"density_windows.png", dpi=180)
    plt.close(fig)
    ap = pd.read_csv(directory/"ap_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(ap.lead_hours, ap.forecast_abs_error, label="Forecast ap q50", color="#7356b3")
    ax.plot(ap.lead_hours, ap.persistence_abs_error, label="Persistence ap", color="#147a88")
    ax.set(xlabel="Forecast lead (hours)", ylabel="ap MAE")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(directory/"ap_accuracy.png", dpi=180)
    plt.close(fig)
    h24 = overall.loc[overall.lead_hours.eq(24)].set_index("model")
    storm24 = storm_scores.loc[storm_scores.lead_hours.eq(24)].set_index("model")
    fixed_under = 100*storm24.loc["dtm2020", "underestimate_gt_30pct"]
    forecast_under = 100*storm24.loc["dtm2020_ap_forecast", "underestimate_gt_30pct"]
    delta = h24.loc["dtm2020_ap_forecast", "mape_pct"]-h24.loc["dtm2020", "mape_pct"]
    result = "снизилась" if delta < 0 else "выросла" if delta > 0 else "не изменилась"
    columns = ["lead_hours", "model", "n", "mape_pct", "bias_pct", "log_rmse", "underestimate_gt_30pct"]
    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda value: f"{value:.3f}")
    limits = "".join(f"<li>{html.escape(text)}</li>" for text in info["limitations"])
    report = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Три варианта прогноза плотности LEO</title>
<style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#202b36;background:#fafbfc}}
table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}h1{{line-height:1.2}}</style>
<h1>JB2008, DTM2020 с фиксированными входами и DTM2020 с прогнозом ap</h1>
<p>Те же три окна 2025 года, GRACE-FO 1, 42 дня, шаг 30 минут. Высоты {data.altitude_km.min():.1f}–{data.altitude_km.max():.1f} км.
Первые два варианта сохранены из исходного эксперимента. В третьем изменяются только геомагнитные драйверы через прогноз ap → Kp.
F10.7 и его среднее совпадают со вторым вариантом.</p>
<p><b>На 24 часах ошибка DTM при добавлении прогноза ap {result} на {abs(delta):.2f} процентного пункта.</b>
MAPE: JB2008 {h24.loc['jb2008','mape_pct']:.2f}%, DTM fixed {h24.loc['dtm2020','mape_pct']:.2f}%,
DTM forecast ap {h24.loc['dtm2020_ap_forecast','mape_pct']:.2f}%. Это результат конкретного экспериментального прогноза ap,
не доказательство преимущества любого прогноза драйверов.</p>
<p><b>Для drag risk нужно отдельно учитывать недооценку во время бурь.</b> При Kp ≥ 5 на 24 часах доля
недооценок плотности более чем на 30% составляет {fixed_under:.1f}% у DTM fixed и {forecast_under:.1f}% у DTM forecast ap.
Средняя ошибка и частота опасно низких оценок могут изменяться в разных направлениях. Медианный прогноз ap
не является защитной верхней границей плотности; текущий результат не калибрует вероятность риска.</p>
<img src="comparison.png" alt="Три модели по горизонту"><h2>Метрики на общих строках</h2>{table(overall[columns])}
<p>MAPE — средняя абсолютная относительная ошибка. underestimate_gt_30pct — доля случаев недооценки плотности более чем на 30%.
Это диагностический показатель, не вероятность потери спутника. Lead 0 — контроль: варианты DTM идентичны.</p>
<h2>Покрытие</h2>{table(coverage)}<p>При недоступности прогноза ap третий вариант не подменяется persistence.
Все варианты оцениваются на одном пересечении строк для каждого горизонта; исходные полные результаты остаются в v1.</p>
<h2>Парные различия MAPE</h2>{table(differences)}<p>Разница = model − reference в процентных пунктах.
Отрицательная величина в пользу model. Интервалы — описательный bootstrap 2000 выборок блоками по UTC-дням;
корреляция соседних штормовых дней не устраняется.</p>
<h2>Геомагнитные условия</h2>{table(scores.loc[scores.grouping.eq('lead_hours+activity'), ['activity',*columns]])}
<p>Quiet: Kp &lt; 3; moderate: 3 ≤ Kp &lt; 5; storm: Kp ≥ 5 в момент проверки.
Будущий фактический Kp используется только для группировки ошибок.</p>
<h2>Контрольные окна</h2><img src="density_windows.png" alt="Плотность в трёх окнах">
<h2>Качество прогноза ap отдельно</h2><img src="ap_accuracy.png" alt="MAE прогноза ap и persistence">
<p>Новая экспериментальная LightGBM q50: {info['training']['training_rows']:,} обучающих строк,
последняя доступная обучающая цель {html.escape(info['training']['max_label_available_at'])}.
Фиксированные параметры до проверки; ни плотность, ни ap за 2025 год не использованы для подбора.
У старой рабочей модели ap нет встроенной границы обучения, поэтому она не использована.
Солнечный ветер, IMF и Dst поступают после закрытия часа, ap после закрытия трёхчасового интервала, F10.7 после закрытия суток.
Расчёт для получасовой даты использует последний закрытый час. Не подставляются будущие фактические солнечный ветер или индексы.</p>
<h2>Ограничения</h2><ul>{limits}</ul>
<h2>Воспроизведение</h2><pre>.venv/bin/python -u scripts/training/thermosphere/train_ap.py
.venv/bin/python -u scripts/training/thermosphere/run_three_way.py
.venv/bin/python -m pytest scripts/training/thermosphere -q</pre>
<p>Сначала должен быть выполнен исходный run.py. Конфигурация — config_three_way.json; результаты и контрольные суммы — results.json;
обучение — ap_training.json; прогнозы ap — ap_forecasts.parquet; прогнозы плотности — predictions.parquet.
Рабочие модели и конфигурация обслуживания не изменены.</p></html>"""
    (directory/"report.html").write_text(report)


if __name__ == "__main__":
    from train_ap import OUT
    write_report(OUT)
