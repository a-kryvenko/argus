"""Generate a standalone Russian HTML report and exportable PNG plots."""
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3]/"data/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_report(directory):
    directory = Path(directory)
    metrics = pd.read_csv(directory / "metrics.csv")
    coverage = pd.read_csv(directory / "prediction_coverage.csv")
    results = json.loads((directory / "results.json").read_text())
    frame = pd.read_parquet(directory / "predictions.parquet").dropna(subset=["jb2008", "dtm2020"])
    overall = metrics.loc[metrics.grouping.eq("lead_hours")]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, label, color in [("jb2008", "Current JB2008", "#d06632"), ("dtm2020", "DTM2020 Operational", "#147a88")]:
        values = overall.loc[overall.model.eq(model)]
        for ax, column, ylabel in [(axes[0], "mape_pct", "Mean absolute percentage error (%)"),
                                   (axes[1], "bias_pct", "Mean relative bias (%)")]:
            ax.plot(values.lead_hours, values[column], "o-", label=label, color=color)
            ax.set(xlabel="Persistence lead (hours)", ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0].legend()
    axes[1].axhline(0, color="grey", linewidth=.8)
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for ax, (window, group) in zip(axes, frame.loc[frame.lead_hours.eq(24)].groupby("window")):
        daily = group.set_index("valid_time")[["truth", "jb2008", "dtm2020"]].resample("6h").mean()*1e12
        for column, label, color in [("truth", "GRACE-FO", "#242b36"), ("jb2008", "Current JB2008", "#d06632"),
                                     ("dtm2020", "DTM2020 Operational", "#147a88")]:
            ax.plot(daily.index, daily[column], label=label, color=color)
        ax.set(title=f"Window {window}; 24 h persistence, 6 h display averages", ylabel="Density (10⁻¹² kg/m³)")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3)
    fig.savefig(directory / "density_windows.png", dpi=180)
    plt.close(fig)
    columns = ["lead_hours", "model", "n", "mape_pct", "bias_pct", "log_rmse", "underestimate_gt_30pct"]
    table = overall[columns].to_html(index=False, float_format=lambda value: f"{value:.3f}", border=0)
    activity = metrics.loc[metrics.grouping.eq("lead_hours+activity"), ["activity", *columns]]
    intervals = pd.DataFrame(results["bootstrap"])
    limitations = "".join(f"<li>{html.escape(text)}</li>" for text in results["limitations"])
    title = "DTM2020 Operational и текущий JB2008 — первый эксперимент"
    h24 = overall.loc[overall.lead_hours.eq(24)].set_index("model").mape_pct
    relative_gain = 100*(1-h24.dtm2020/h24.jb2008)
    output = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>{title}</title>
<style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#202b36;background:#fafbfc}}
h1{{line-height:1.2}}table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}code{{background:#eee;padding:3px}}a{{color:#147a88}}</style>
<h1>{title}</h1><p>42 дня 2025 года; измерения GRACE-FO 1, высоты {frame.altitude_km.min():.1f}–{frame.altitude_km.max():.1f} км.
Общие даты и координаты, одинаковый режим persistence, горизонты 0/6/12/24/48 часов.
Победитель определяется только по общим строкам с доступными прогнозами обеих моделей.</p>
<p><b>На 24 часах MAPE: DTM2020 {h24.dtm2020:.1f}%, JB2008 {h24.jb2008:.1f}%</b> —
изменение ошибки на {relative_gain:.1f}% в пользу DTM в этой выборке. Это основание для расширения проверки,
а не для автоматической замены модели в продукте. Проверяется вся текущая цепочка драйверов JB2008,
поэтому разницу нельзя приписать только формулам атмосферных моделей.</p>
<p><b>Это сравнение плотности, а не проверка риска потери спутника.</b> Использованы архивные измерения, без переобучения,
коррекции систематического смещения и подгонки под эталон. Предстоящая орбита взята из наблюдений для обеих моделей.</p>
<img src="comparison.png" alt="Ошибки моделей по горизонту"><h2>Парные метрики</h2>{table}
<p>MAPE — средняя абсолютная относительная ошибка. Bias — средняя относительная ошибка со знаком.
Доля underestimate_gt_30pct показывает недооценку плотности более чем на 30%; это диагностический порог, не калиброванный уровень риска.</p>
<h2>Доступность входов и прогнозов</h2>{coverage.to_html(index=False, border=0)}
<p>Ошибки и причины пропусков сохранены для каждой строки в predictions.parquet. Пропуски JB2008 не исключаются из отчёта покрытия.</p>
<h2>Разница ошибок и описательный интервал</h2>{intervals.to_html(index=False, float_format=lambda v: f'{v:.3f}', border=0)}
<p>Разница = MAPE DTM − MAPE JB в процентных пунктах; отрицательная величина в пользу DTM.
2000 повторных выборок блоками по UTC-дням; зависимость соседних дней одной бури ограничивает интерпретацию интервалов.</p>
<h2>Плотность на горизонте 24 часа</h2><img src="density_windows.png" alt="Плотность по трём окнам">
<h2>Геомагнитные условия в момент проверки</h2><p>Quiet: Kp &lt; 3; moderate: 3 ≤ Kp &lt; 5; storm: Kp ≥ 5.
Будущий Kp используется только для группировки результатов, не для прогнозирования.</p>{activity.to_html(index=False, float_format=lambda v:f'{v:.3f}', border=0)}
<h2>Протокол и ограничения</h2><ul>{limitations}</ul>
<p>JB2008 использует исходные функции рабочего backend: реконструкцию S10/M10/Y10 из GOES, trailing-средние,
causal DTC и фиксацию драйверов. DTM2020 использует исходную функцию dtm3 без изменений, отдельный интерфейс
и штатные коэффициенты. Четыре опубликованные контрольные точки проверены с rtol=5e-5.</p>
<p>В исходном standalone lecdtm есть неопределённая переменная iu10. Наш интерфейс читает неизменённый файл коэффициентов
по его фиксированному формату и инициализирует COMMON перед вызовом неизменённой dtm3; исходник не исправлялся.</p>
<h2>Повторение</h2><pre>.venv/bin/python scripts/training/thermosphere/prepare.py
.venv/bin/python -u scripts/training/thermosphere/run.py
.venv/bin/python -m pytest scripts/training/thermosphere/test_experiment.py</pre>
<p>Требуются локальный private forecast-core, существующие архивы OMNI/GOES, frozen calibration,
cdflib, Astropy, pyatmos, pandas, NumPy, Matplotlib и gfortran. Сетевой доступ нужен только prepare.py.
Конфигурация: scripts/training/thermosphere/config.json. Контрольные суммы: download_manifest.json, build.json, results.json.
Измерения TU Delft/ESA: <a href="https://swarm-diss.eo.esa.int/#swarm/Multimission/GRACE-FO/DNS/Sat_1">архив GRACE-FO</a>;
модель: <a href="https://swami-h2020-eu.github.io/mcm/dtm2020.html">DTM2020 / SWAMI</a>.</p>
</html>"""
    (directory / "report.html").write_text(output)


if __name__ == "__main__":
    from prepare import OUT
    write_report(OUT)
