const numberFormatter = new Intl.NumberFormat('es-ES', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const percentFormatter = new Intl.NumberFormat('es-ES', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

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
  setText('report-roi', report.roi || '-');
  setText('report-sharpe', report.sharpe || '-');
  setText('report-pf', report.profit_factor || '-');

  const button = document.getElementById('download-report');
  if (button) {
    button.onclick = () => {
      const blob = new Blob([report.chart_csv || ''], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'synthetic_daily_report.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    };
  }
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
  } catch (error) {
    console.error('Error actualizando dashboard', error);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  refreshDashboard();
  setInterval(refreshDashboard, 5000);
});
