const numberFormatter = new Intl.NumberFormat('es-ES', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const percentFormatter = new Intl.NumberFormat('es-ES', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const allowedModes = new Set(['backtest', 'paper', 'live']);

let latestReportCsv = '';

function setFeedbackText(id, message, isError = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message;
  el.classList.toggle('error', Boolean(isError));
}

function formatValue(value, fallback = '-') {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return fallback;
  }
  return numberFormatter.format(value);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-';
  }
  return percentFormatter.format(value);
}

function setText(id, value, formatter = (v) => v) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = formatter(value);
}

function applyStatusClass(cell, value) {
  if (!cell) return;
  cell.classList.remove('status-positive', 'status-negative');
  if (typeof value !== 'number') {
    return;
  }
  if (value > 0) {
    cell.classList.add('status-positive');
  } else if (value < 0) {
    cell.classList.add('status-negative');
  }
}

async function fetchDashboard() {
  const response = await fetch('/dashboard/data');
  if (!response.ok) {
    throw new Error(`Error HTTP ${response.status}`);
  }
  return response.json();
}

function updateStatus(status) {
  if (!status) return;
  setText('status-equity', status.equity, (v) => `${formatValue(v)} USD`);
  setText('status-dd', status.daily_dd, (v) => `${formatValue(v)} USD`);
  setText('status-positions', status.positions);
  setText('status-mode', status.mode || '-');

  const modeSelect = document.getElementById('mode-select');
  if (modeSelect && typeof status.mode === 'string' && allowedModes.has(status.mode)) {
    modeSelect.value = status.mode;
  }
}

function updateTradeSummary(summary) {
  const defaults = {
    closed_trades: '-',
    open_trades: '-',
    wins: '-',
    losses: '-',
    win_rate: '-',
    loss_rate: '-',
    avg_win: '-',
    avg_loss: '-',
    best_trade: '-',
    worst_trade: '-',
    total_pnl: '-',
  };

  const payload = summary || defaults;

  setText('summary-closed', payload.closed_trades);
  setText('summary-open', payload.open_trades);
  setText('summary-wins', payload.wins);
  setText('summary-losses', payload.losses);
  setText('summary-winrate', payload.win_rate, formatPercent);
  setText('summary-lossrate', payload.loss_rate, formatPercent);
  setText('summary-avgwin', payload.avg_win, formatValue);
  setText('summary-avgloss', payload.avg_loss, formatValue);
  setText('summary-best', payload.best_trade, formatValue);
  setText('summary-worst', payload.worst_trade, formatValue);
  setText('summary-totalpnl', payload.total_pnl, formatValue);

  applyStatusClass(document.getElementById('summary-totalpnl'), payload.total_pnl);
  applyStatusClass(document.getElementById('summary-avgwin'), payload.avg_win);
  applyStatusClass(document.getElementById('summary-avgloss'), payload.avg_loss);
  applyStatusClass(document.getElementById('summary-best'), payload.best_trade);
  applyStatusClass(document.getElementById('summary-worst'), payload.worst_trade);
}

function updateReport(report) {
  if (!report) return;
  const metrics = report.metrics || {};
  setText('report-roi', metrics.roi, formatPercent);
  setText('report-sharpe', metrics.sharpe, formatValue);
  setText('report-pf', metrics.profit_factor, formatValue);
  setText('report-winrate', metrics.win_rate, formatPercent);
  setText('report-dd', metrics.max_drawdown, formatPercent);
  applyStatusClass(document.getElementById('report-roi'), metrics.roi);
  applyStatusClass(document.getElementById('report-pf'), metrics.profit_factor);
  applyStatusClass(document.getElementById('report-winrate'), metrics.win_rate);
  latestReportCsv = report.chart_csv || '';
}

function updateTradesTable(trades) {
  const tbody = document.querySelector('#trades-table tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  (trades || []).forEach((trade) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${trade.symbol || '-'}</td>
      <td>${trade.entry_price !== null && trade.entry_price !== undefined ? formatValue(trade.entry_price) : '-'}</td>
      <td>${trade.exit_price !== null && trade.exit_price !== undefined ? formatValue(trade.exit_price) : '-'}</td>
      <td>${trade.quantity !== null && trade.quantity !== undefined ? formatValue(trade.quantity) : '-'}</td>
      <td class="${trade.pnl > 0 ? 'status-positive' : trade.pnl < 0 ? 'status-negative' : ''}">${
        trade.pnl !== null && trade.pnl !== undefined ? formatValue(trade.pnl) : '-'
      }</td>
      <td>${trade.opened_at ? new Date(trade.opened_at).toLocaleString() : '-'}</td>
      <td>${trade.closed_at ? new Date(trade.closed_at).toLocaleString() : '-'}</td>
    `;
    tbody.appendChild(row);
  });
}

function updateDailyPnlTable(items) {
  const tbody = document.querySelector('#pnl-table tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  (items || []).forEach((item) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${item.day ? new Date(item.day).toLocaleDateString() : '-'}</td>
      <td class="${item.pnl > 0 ? 'status-positive' : item.pnl < 0 ? 'status-negative' : ''}">${
        item.pnl !== null && item.pnl !== undefined ? formatValue(item.pnl) : '-'
      }</td>
      <td>${item.trades ?? '-'}</td>
    `;
    tbody.appendChild(row);
  });
}

async function refreshDashboard() {
  try {
    const data = await fetchDashboard();
    updateStatus(data.status);
    updateTradeSummary(data.trade_summary);
    updateReport(data.report);
    updateTradesTable(data.recent_trades);
    updateDailyPnlTable(data.daily_pnl);
    setFeedbackText('report-feedback', '');
  } catch (error) {
    console.error('Error actualizando dashboard', error);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  refreshDashboard();
  setInterval(refreshDashboard, 5000);

  const modeSelect = document.getElementById('mode-select');
  const modeButton = document.getElementById('mode-apply');
  const feedback = document.getElementById('mode-feedback');

  const setFeedback = (message, isError = false) => {
    if (!feedback) return;
    feedback.textContent = message;
    feedback.classList.toggle('mode-feedback-error', Boolean(isError));
  };

  if (modeButton && modeSelect) {
    modeButton.addEventListener('click', async () => {
      const mode = modeSelect.value;
      setFeedback('');
      try {
        const response = await fetch('/mode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode }),
        });

        if (response.status === 400 || response.status === 403) {
          const errorData = await response.json().catch(() => ({ detail: 'Error actualizando modo' }));
          setFeedback(errorData.detail || 'No se pudo actualizar el modo', true);
          return;
        }

        if (!response.ok) {
          throw new Error(`Error HTTP ${response.status}`);
        }

        const data = await response.json();
        if (data.mode) {
          setText('status-mode', data.mode);
          if (allowedModes.has(data.mode)) {
            modeSelect.value = data.mode;
          }
          setFeedback(`Modo actualizado a ${data.mode}`);
        } else {
          setFeedback('Modo actualizado');
        }
        refreshDashboard();
      } catch (error) {
        console.error('Error cambiando modo operativo', error);
        setFeedback('Error inesperado al cambiar el modo', true);
      }
    });
  }

  const backtestForm = document.getElementById('backtest-form');
  if (backtestForm) {
    backtestForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const symbol = document.getElementById('backtest-symbol')?.value ?? '';
      const timeframe = document.getElementById('backtest-timeframe')?.value ?? '';
      const start = document.getElementById('backtest-start')?.value ?? '';
      const end = document.getElementById('backtest-end')?.value ?? '';
      setFeedbackText('backtest-feedback', 'Ejecutando backtest…');
      try {
        const response = await fetch('/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, timeframe, start, end }),
        });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Error ejecutando backtest' }));
          setFeedbackText('backtest-feedback', errorData.detail || 'No se pudo ejecutar el backtest', true);
          return;
        }
        const result = await response.json();
        const metrics = result.metrics || {};
        setText('backtest-roi', metrics.roi, formatPercent);
        setText('backtest-sharpe', metrics.sharpe, formatValue);
        setText('backtest-pf', metrics.profit_factor, formatValue);
        setText('backtest-winrate', metrics.win_rate, formatPercent);
        setText('backtest-dd', metrics.max_drawdown, formatPercent);
        setText('backtest-ready', result.go_live_ready === undefined ? '-' : result.go_live_ready ? 'Sí' : 'No');
        applyStatusClass(document.getElementById('backtest-roi'), metrics.roi);
        applyStatusClass(document.getElementById('backtest-pf'), metrics.profit_factor);
        applyStatusClass(document.getElementById('backtest-winrate'), metrics.win_rate);
        setFeedbackText('backtest-feedback', 'Backtest completado correctamente');
      } catch (error) {
        console.error('Error ejecutando backtest', error);
        setFeedbackText('backtest-feedback', 'Error inesperado en el backtest', true);
      }
    });
  }

  const guardButton = document.getElementById('guard-run');
  if (guardButton) {
    guardButton.addEventListener('click', async () => {
      setFeedbackText('guard-feedback', 'Evaluando métricas…');
      try {
        const response = await fetch('/live/guard', { method: 'POST' });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Validación no disponible' }));
          setFeedbackText('guard-feedback', errorData.detail || 'No se pudo validar', true);
          return;
        }
        const result = await response.json();
        const metrics = result.metrics || {};
        setText('guard-roi', metrics.roi, formatPercent);
        setText('guard-sharpe', metrics.sharpe, formatValue);
        setText('guard-pf', metrics.profit_factor, formatValue);
        setText('guard-winrate', metrics.win_rate, formatPercent);
        setText('guard-dd', metrics.max_drawdown, formatPercent);
        setText('guard-ready', result.go_live_ready === undefined ? '-' : result.go_live_ready ? 'Sí' : 'No');
        applyStatusClass(document.getElementById('guard-roi'), metrics.roi);
        applyStatusClass(document.getElementById('guard-pf'), metrics.profit_factor);
        applyStatusClass(document.getElementById('guard-winrate'), metrics.win_rate);
        setFeedbackText('guard-feedback', result.go_live_ready ? 'Listo para operar en vivo' : 'Métricas insuficientes');
      } catch (error) {
        console.error('Error validando live trading', error);
        setFeedbackText('guard-feedback', 'Error inesperado al validar', true);
      }
    });
  }

  const reportRefresh = document.getElementById('report-refresh');
  if (reportRefresh) {
    reportRefresh.addEventListener('click', async () => {
      setFeedbackText('report-feedback', 'Generando reporte diario…');
      try {
        const response = await fetch('/report/daily');
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'No se pudo generar el reporte' }));
          setFeedbackText('report-feedback', errorData.detail || 'No se pudo generar el reporte', true);
          return;
        }
        const data = await response.json();
        updateReport(data);
        setFeedbackText('report-feedback', 'Reporte actualizado');
      } catch (error) {
        console.error('Error generando reporte diario', error);
        setFeedbackText('report-feedback', 'Error inesperado generando reporte', true);
      }
    });
  }

  const downloadButton = document.getElementById('download-report');
  if (downloadButton) {
    downloadButton.addEventListener('click', () => {
      const blob = new Blob([latestReportCsv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'synthetic_daily_report.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    });
  }
});
