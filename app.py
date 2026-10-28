import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="IDX Equity Valuation Analytics",
    page_icon="📈",
    layout="wide",
)

st.title("IDX Equity Valuation Analytics")
st.markdown(
    "Analisis Valuasi Komprehensif Saham BEI Menggunakan Metode DCF-CAPM, Relative Valuation, dan DDM Berbasis Margin of Safety."
)


def _latest_value_for_labels(balance_sheet, labels):
  """Ambil nilai dari akun yang cocok persis pada periode neraca terbaru."""
  if balance_sheet is None or balance_sheet.empty:
    return None, None, None

  # Kolom yfinance biasanya tanggal akhir periode; pilih tanggal terbaru secara eksplisit.
  columns = sorted(balance_sheet.columns, key=lambda x: pd.Timestamp(x), reverse=True)
  normalized_index = {str(idx).strip().lower(): idx for idx in balance_sheet.index}

  for label in labels:
    row = normalized_index.get(label.strip().lower())
    if row is None:
      continue
    for col in columns:
      value = balance_sheet.loc[row, col]
      if isinstance(value, pd.Series):
        value = value.iloc[0]
      if pd.notna(value):
        try:
          return float(value), str(col), str(row)
        except (TypeError, ValueError):
          continue
  return None, None, None


def _get_cash_and_debt(balance_sheet):
  """Ambil kas dan utang dari label neraca yang spesifik, tanpa mencampur periode."""
  cash, period, cash_label = _latest_value_for_labels(
      balance_sheet,
      ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"],
  )

  debt, debt_period, debt_label = _latest_value_for_labels(
      balance_sheet,
      ["Total Debt", "TotalDebt"],
  )

  # Jika total debt tidak tersedia, jumlahkan hanya akun pinjaman jangka pendek
  # dan panjang yang eksplisit. Jangan gunakan total liabilities sebagai utang.
  if debt is None:
    current_debt, current_period, current_label = _latest_value_for_labels(
        balance_sheet,
        ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Long Term Debt"],
    )
    long_debt, long_period, long_label = _latest_value_for_labels(
        balance_sheet,
        ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    )
    if current_debt is not None or long_debt is not None:
      debt = (current_debt or 0.0) + (long_debt or 0.0)
      debt_period = current_period or long_period
      debt_label = "Penjumlahan utang lancar + utang jangka panjang"

  # Kas dan utang wajib berasal dari tanggal laporan yang sama.
  # Bila akun ditemukan pada periode berbeda, tandai agar tidak digunakan diam-diam.
  if cash is not None and debt is not None and period and debt_period and period != debt_period:
    return cash, debt, {"cash_period": period, "debt_period": debt_period,
                        "cash_label": cash_label, "debt_label": debt_label,
                        "period_mismatch": True}
  return cash, debt, {"cash_period": period, "debt_period": debt_period,
                      "cash_label": cash_label, "debt_label": debt_label,
                      "period_mismatch": False}


@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
  try:
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    # Pisahkan mata uang harga saham dari mata uang laporan keuangan.
    # Untuk saham BEI, harga dapat dalam IDR sementara laporan keuangan emiten
    # (misalnya BUMI) disajikan dalam USD.
    currency = info.get("currency", "IDR")
    financial_currency = info.get("financialCurrency", currency)

    # Ambil data historis keuangan
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow

    # Harga terkini & Market Cap
    current_price = info.get(
        "currentPrice",
        info.get("regularMarketPrice", info.get("previousClose", 0)),
    )
    shares_outstanding = info.get("sharesOutstanding", 0)
    market_cap = info.get("marketCap", current_price * shares_outstanding)

    # Kas dan utang: gunakan label akun yang spesifik dan periode terbaru.
    total_cash, total_debt, balance_meta = _get_cash_and_debt(balance_sheet)
    if total_cash is None:
      total_cash = 0.0
    if total_debt is None:
      total_debt = 0.0

    # FCF historis: validasi angka Yahoo Finance terhadap arus kas operasi.
    # Jangan menerima FCF yang jauh di luar skala OCF; jika tidak valid, hitung
    # dari OCF + capex hanya bila kedua baris agregat tersedia dan cocok periodenya.
    fcf_list = []
    fcf_records = []
    fcf_source = "Tidak ditemukan / belum tervalidasi"
    if cashflow is not None and not cashflow.empty:
      cf_columns = sorted(cashflow.columns, key=lambda x: pd.Timestamp(x), reverse=True)

      def _norm_label(value):
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())

      normalized_cf = {_norm_label(idx): idx for idx in cashflow.index}
      direct_fcf_row = next(
        (normalized_cf[k] for k in ["freecashflow", "freecashflowcontinuing"] if k in normalized_cf),
        None,
      )
      ocf_row = next(
        (normalized_cf[k] for k in [
          "operatingcashflow",
          "cashflowfromcontinuingoperatingactivities",
          "totalcashfromoperatingactivities",
          "operatingcashflowcontinuing",
          "cashflowsfromusedinoperatingactivitiesdirect",
          "cashflowsfromusedinoperatingactivities",
          "cashflowsfromoperatingactivities",
        ] if k in normalized_cf),
        None,
      )
      # Yahoo Finance sometimes returns localized/expanded labels for the aggregate
      # operating cash flow. Use a narrow fallback only if exact aliases were absent.
      if ocf_row is None:
        for normalized_label, original_label in normalized_cf.items():
          if (
            "cashflowsfromusedinoperatingactivities" in normalized_label
            or "cashflowsfromoperatingactivities" in normalized_label
          ):
            ocf_row = original_label
            break
      capex_row = next(
        (normalized_cf[k] for k in [
          "capitalexpenditure",
          "capitalexpenditures",
          "capitalexpenditurereported",
          "capitalexpendituresreported",
          "purchaseofppe",
          "purchaseofpropertyplantequipment",
        ] if k in normalized_cf),
        None,
      )

      # Buat diagnostik per periode; simpan angka mentah yang membantu audit.
      for col in cf_columns:
        record = {"Periode": str(pd.Timestamp(col).date())}
        direct_value = cashflow.loc[direct_fcf_row, col] if direct_fcf_row is not None else np.nan
        ocf_value = cashflow.loc[ocf_row, col] if ocf_row is not None else np.nan
        capex_value = cashflow.loc[capex_row, col] if capex_row is not None else np.nan
        record["FCF Yahoo mentah (USD)"] = float(direct_value) if pd.notna(direct_value) else None
        record["OCF agregat (USD)"] = float(ocf_value) if pd.notna(ocf_value) else None
        record["CapEx (USD)"] = float(capex_value) if pd.notna(capex_value) else None

        accepted_value = None
        status = "Tidak dihitung: baris agregat OCF/CapEx tidak tersedia"
        if pd.notna(direct_value):
          direct_value = float(direct_value)
          # FCF puluhan/triliunan USD tidak wajar untuk emiten ini jika OCF hanya puluhan juta.
          if pd.notna(ocf_value):
            ocf_float = float(ocf_value)
            plausible = abs(direct_value) <= max(abs(ocf_float) * 10, 1_000_000_000)
          else:
            plausible = abs(direct_value) <= 10_000_000_000
          if plausible:
            accepted_value = direct_value
            status = "FCF Yahoo diterima; skala relatif terhadap OCF lolos pemeriksaan awal"
          elif pd.notna(ocf_value) and pd.notna(capex_value):
            calculated = float(ocf_value) + float(capex_value)
            # Tolak hasil turunan juga jika jauh dari skala OCF tanpa penjelasan.
            if abs(calculated) <= max(abs(float(ocf_value)) * 10, 1_000_000_000):
              accepted_value = calculated
              status = "FCF Yahoo ditolak sebagai outlier; memakai OCF + CapEx"
            else:
              status = "FCF Yahoo dan hasil OCF + CapEx tidak lolos pemeriksaan skala"
          else:
            status = "FCF Yahoo ditolak sebagai outlier; OCF/CapEx agregat tidak cukup untuk menghitung ulang"
        elif pd.notna(ocf_value) and pd.notna(capex_value):
          calculated = float(ocf_value) + float(capex_value)
          if abs(calculated) <= max(abs(float(ocf_value)) * 10, 1_000_000_000):
            accepted_value = calculated
            status = "Dihitung dari OCF + CapEx"
          else:
            status = "Hasil OCF + CapEx tidak lolos pemeriksaan skala"

        record["FCF tervalidasi (USD)"] = accepted_value
        record["Status pemeriksaan"] = status
        fcf_records.append(record)
        # DCF hanya memakai FCF dari periode paling baru; jangan diam-diam
        # mengganti tahun dasar dengan tahun lama jika periode terbaru invalid.
        if accepted_value is not None and col == cf_columns[0]:
          fcf_list.append(accepted_value)

      if direct_fcf_row is not None:
        fcf_source = f"Yahoo Finance: {direct_fcf_row} (diperiksa terhadap skala OCF)"
      elif ocf_row is not None and capex_row is not None:
        fcf_source = f"Dihitung dari {ocf_row} + {capex_row}"
    # Dividen history untuk DDM
    dividends = stock.dividends

    return {
        "info": info,
        "currency": currency,
        "financial_currency": financial_currency,
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "balance_meta": balance_meta,
        "fcf_list": fcf_list,
        "fcf_records": fcf_records,
        "fcf_source": fcf_source,
        "ocf_label_detected": str(ocf_row) if ocf_row is not None else None,
        "capex_label_detected": str(capex_row) if capex_row is not None else None,
        "dividends": dividends,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
    }, None
  except Exception as e:
    return None, str(e)


# Input Ticker di Sidebar
st.sidebar.header("🔍 Pencarian Emiten")
user_ticker = (
    st.sidebar.text_input("Masukkan Ticker IDX (Contoh: BBCA, ASII, ARCI)", "BBCA")
    .upper()
    .strip()
)
if not user_ticker.endswith(".JK") and user_ticker != "":
  ticker_full = user_ticker + ".JK"
else:
  ticker_full = user_ticker

data, err = get_stock_data(ticker_full)

# Ambil Kurs USDIDR jika emiten USD
usd_idr = 16000  # Fallback default
try:
  fx = yf.Ticker("USDIDR=X").info
  usd_idr = fx.get("regularMarketPrice", fx.get("previousClose", 16000))
except:
  pass

if err or not data or data["current_price"] == 0:
  st.error(
      f"Gagal memuat data untuk ticker {ticker_full}. Pastikan kode saham benar"
      " dan terdaftar di BEI."
  )
else:
  info = data["info"]
  currency = data["currency"]
  financial_currency = data.get("financial_currency", currency)
  # Multiplier umum untuk harga/per-share mengikuti mata uang quote.
  fx_multiplier = usd_idr if currency == "USD" else 1.0
  # Multiplier khusus data neraca dan arus kas mengikuti mata uang laporan.
  financial_fx_multiplier = usd_idr if financial_currency == "USD" else 1.0

  st.sidebar.success(
      f"Berhasil memuat: {info.get('longName', ticker_full)} ({currency})"
  )
  # Transparansi sumber akun neraca untuk membantu memvalidasi angka.
  balance_meta = data.get("balance_meta", {})
  if balance_meta.get("period_mismatch"):
    st.sidebar.warning(
        "Periode kas dan utang tidak sama. Periksa neraca sebelum menggunakan hasil valuasi. "
        f"Kas: {balance_meta.get('cash_period')}; utang: {balance_meta.get('debt_period')}"
    )
  with st.sidebar.expander("Detail sumber kas & utang"):
    st.write("Mata uang harga saham:", currency)
    st.write("Mata uang laporan keuangan:", financial_currency)
    st.write("Kurs USD/IDR yang digunakan:", f"Rp {usd_idr:,.2f}" if financial_currency == "USD" else "Tidak diperlukan")
    st.write("Akun kas:", balance_meta.get("cash_label") or "Tidak ditemukan")
    st.write("Periode kas:", balance_meta.get("cash_period") or "Tidak tersedia")
    st.write("Akun utang:", balance_meta.get("debt_label") or "Tidak ditemukan")
    st.write("Periode utang:", balance_meta.get("debt_period") or "Tidak tersedia")
    if not balance_meta.get("cash_label") or not balance_meta.get("debt_label"):
      st.warning("Salah satu akun tidak ditemukan di Yahoo Finance; nilai 0 bukan berarti saldo sebenarnya nol.")

  # Diagnostik sumber FCF: tampilkan akun arus kas mentah agar bisa memastikan
  # apakah FCF nol karena akun tidak dikenali atau memang nilai historisnya nol/negatif.
  with st.sidebar.expander("Diagnostik sumber FCF (arus kas)"):
    cashflow_df = data.get("cashflow")
    st.write("Mata uang laporan arus kas:", financial_currency)
    if cashflow_df is None or cashflow_df.empty:
      st.warning("Yahoo Finance tidak mengembalikan tabel arus kas untuk emiten ini.")
    else:
      st.write("Baris akun yang tersedia di Yahoo Finance:")
      st.dataframe(cashflow_df, use_container_width=True)
      st.write("Nama akun lengkap (untuk pemeriksaan pencocokan):")
      st.code("\\n".join(
        f"ASLI: {repr(idx)} | NORMAL: {''.join(ch.lower() for ch in str(idx) if ch.isalnum())}"
        for idx in cashflow_df.index
      ))
      st.write("Baris OCF yang berhasil dicocokkan:", data.get("ocf_label_detected") or "TIDAK DITEMUKAN")
      st.write("Baris CapEx yang berhasil dicocokkan:", data.get("capex_label_detected") or "TIDAK DITEMUKAN")
      st.write("Sumber FCF yang digunakan:", data.get("fcf_source", "Tidak diketahui"))
      st.write("Pemeriksaan FCF per periode (sebelum konversi kurs):")
      fcf_records = data.get("fcf_records", [])
      if fcf_records:
        st.dataframe(pd.DataFrame(fcf_records), use_container_width=True, hide_index=True)
      else:
        st.warning("Tidak ada nilai FCF historis yang berhasil dibaca.")
      st.caption("FCF mentah yang tidak konsisten dengan skala OCF ditolak; FCF tervalidasi saja yang dapat masuk ke perhitungan DCF.")

  # --- SIDEBAR: ASUMSI VALUASI ---
  st.sidebar.markdown("---")
  st.sidebar.caption("💡 Gunakan asumsi standar, atau buka panel di bawah untuk mengubah asumsi valuasi.")
  with st.sidebar.expander("⚙️ Pengaturan Asumsi Valuasi (DCF)", expanded=False):

    # 1. Parameter WACC (CAPM Components)
    st.subheader("A. Komponen WACC (CAPM)")
    risk_free_rate = (
        st.number_input(
            "Risk-Free Rate Rf (%)", min_value=0.0, max_value=20.0, value=6.5, step=0.1
        )
        / 100
    )
    tax_rate = (
        st.number_input(
            "Tax Rate Perusahaan (%)",
            min_value=0.0,
            max_value=50.0,
            value=22.0,
            step=0.5,
        )
        / 100
    )
    market_risk_premium = (
        st.number_input(
            "Market Risk Premium (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.0,
            step=0.1,
        )
        / 100
    )
    beta = st.number_input(
        "Beta Saham", min_value=0.0, max_value=3.0, value=1.0, step=0.05
    )
    cost_of_debt = (
        st.number_input(
            "Cost of Debt Sebelum Pajak (%)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
        )
        / 100
    )
    debt_weight = (
        st.slider(
            "Bobot Utang / Capital Structure Debt (%)", 0, 100, 20, 5
        )
        / 100
    )
    equity_weight = 1.0 - debt_weight

    # Hitung WACC otomatis
    cost_of_equity = risk_free_rate + (beta * market_risk_premium)
    effective_cost_of_debt = cost_of_debt * (1 - tax_rate)
    calculated_wacc = (equity_weight * cost_of_equity) + (
        debt_weight * effective_cost_of_debt
    )

    st.info(
        f"**Cost of Equity (Ke):** {cost_of_equity*100:.2f}%\n\n"
        f"**Cost of Debt (After Tax):** {effective_cost_of_debt*100:.2f}%\n\n"
        f"⚡ **Total WACC Otomatis:** **{calculated_wacc*100:.2f}%**"
    )

    # 2. Parameter DCF Lainnya
    st.subheader("B. Parameter Proyeksi DCF")
    projection_years = st.selectbox(
        "Time Period Cash Flow", options=[5, 10], index=0
    )
    fcf_growth_rate = (
        st.slider(
            "Pertumbuhan FCF Tahunan (%)", -10.0, 30.0, 8.0, 0.5
        )
        / 100
    )
    perpetual_growth = (
        st.slider("Perpetual Growth Rate (%)", 0.0, 5.0, 2.5, 0.25)
        / 100
    )

    # Proyeksi pemulihan FCF otomatis: asumsi dapat diubah pengguna.
    st.subheader("C. Asumsi Pemulihan FCF Otomatis")
    enable_recovery_scenario = st.checkbox(
        "Aktifkan proyeksi pemulihan FCF", value=True,
        help="Jika FCF historis negatif, model memproyeksikan pendapatan dan margin FCF. Ini skenario, bukan kepastian."
    )
    revenue_growth_assumption = st.slider(
        "Pertumbuhan pendapatan tahunan (%)", -20.0, 40.0, 5.0, 0.5,
        disabled=not enable_recovery_scenario
    ) / 100
    recovery_year = st.selectbox(
        "Margin FCF mencapai target pada tahun ke-", options=[2, 3, 4, 5], index=1,
        disabled=not enable_recovery_scenario,
    )
    target_fcf_margin_pct = st.number_input(
        "Target margin FCF pada tahun pemulihan (%)",
        min_value=-50.0, max_value=50.0, value=5.0, step=0.5,
        help="Target margin FCF = FCF / pendapatan. Gunakan asumsi yang didukung analisis bisnis.",
        disabled=not enable_recovery_scenario,
    )
    target_fcf_margin = target_fcf_margin_pct / 100

    # 4. Margin of Safety
    st.subheader("C. Margin of Safety (MoS)")
    mos_percentage = (
        st.selectbox(
            "Pilih Margin of Safety",
            options=[10, 15, 20, 25, 30, 35, 40, 45, 50],
            index=3,
        )
        / 100
    )

  # --- MAIN DATA PROCESSING ---
  current_price = data["current_price"]
  shares = data["shares_outstanding"]
  total_debt = data["total_debt"] * financial_fx_multiplier
  total_cash = data["total_cash"] * financial_fx_multiplier
  fcf_list = data["fcf_list"]

  # DCF hanya dihitung jika FCF historis tersedia dan positif.
  # Jangan mengganti FCF negatif/kosong dengan angka buatan karena dapat menyesatkan valuasi.
  base_fcf = fcf_list[0] * financial_fx_multiplier if len(fcf_list) > 0 else None
  dcf_valid = (
      base_fcf is not None
      and base_fcf > 0
      and calculated_wacc > perpetual_growth
      and shares > 0
  )

  projected_fcfs = []
  pv_fcfs = []
  if dcf_valid:
    current_fcf = base_fcf
    for yr in range(1, projection_years + 1):
      current_fcf *= 1 + fcf_growth_rate
      projected_fcfs.append(current_fcf)
      pv_fcfs.append(current_fcf / ((1 + calculated_wacc) ** yr))
    sum_pv_fcf = sum(pv_fcfs)
    terminal_value = (projected_fcfs[-1] * (1 + perpetual_growth)) / (calculated_wacc - perpetual_growth)
    pv_terminal_value = terminal_value / ((1 + calculated_wacc) ** projection_years)
    enterprise_value = sum_pv_fcf + pv_terminal_value
    equity_value = enterprise_value + total_cash - total_debt
    intrinsic_value_dcf = equity_value / shares
    target_buy_price = intrinsic_value_dcf * (1 - mos_percentage)
  else:
    sum_pv_fcf = pv_terminal_value = enterprise_value = equity_value = None
    intrinsic_value_dcf = target_buy_price = None
    projected_fcfs = [None] * projection_years
    pv_fcfs = [None] * projection_years

  # DCF pemulihan otomatis berbasis pendapatan dan margin FCF.
  recovery_valid = False
  recovery_projected_fcfs = []
  recovery_pv_fcfs = []
  recovery_intrinsic_value = None
  recovery_target_price = None
  recovery_sum_pv_fcf = None
  recovery_pv_terminal = None
  recovery_ev = None
  recovery_equity_value = None
  latest_fcf_reporting_ccy = fcf_list[0] if fcf_list else None
  revenue_reporting_ccy = float(info.get("totalRevenue") or 0)
  recovery_records = []

  if (
      enable_recovery_scenario and shares > 0 and revenue_reporting_ccy > 0
      and latest_fcf_reporting_ccy is not None
      and calculated_wacc > perpetual_growth and recovery_year <= projection_years
  ):
    starting_margin = latest_fcf_reporting_ccy / revenue_reporting_ccy
    for yr in range(1, projection_years + 1):
      projected_revenue = revenue_reporting_ccy * ((1 + revenue_growth_assumption) ** yr)
      if yr <= recovery_year:
        progress = yr / recovery_year
        projected_margin = starting_margin + (target_fcf_margin - starting_margin) * progress
      else:
        projected_margin = target_fcf_margin * ((1 + fcf_growth_rate) ** (yr - recovery_year))
      fcf_reporting_ccy = projected_revenue * projected_margin
      fcf_idr = fcf_reporting_ccy * financial_fx_multiplier
      recovery_projected_fcfs.append(fcf_idr)
      recovery_pv_fcfs.append(fcf_idr / ((1 + calculated_wacc) ** yr))
      recovery_records.append({
          "Tahun": yr,
          f"Pendapatan proyeksi ({financial_currency})": projected_revenue,
          "Margin FCF proyeksi (%)": projected_margin * 100,
          f"FCF proyeksi ({financial_currency})": fcf_reporting_ccy,
          "FCF proyeksi (IDR)": fcf_idr,
          "PV FCF (IDR)": recovery_pv_fcfs[-1],
      })
    terminal_fcf = recovery_projected_fcfs[-1]
    if terminal_fcf > 0:
      recovery_sum_pv_fcf = sum(recovery_pv_fcfs)
      recovery_terminal_value = (terminal_fcf * (1 + perpetual_growth)) / (calculated_wacc - perpetual_growth)
      recovery_pv_terminal = recovery_terminal_value / ((1 + calculated_wacc) ** projection_years)
      recovery_ev = recovery_sum_pv_fcf + recovery_pv_terminal
      recovery_equity_value = recovery_ev + total_cash - total_debt
      recovery_intrinsic_value = recovery_equity_value / shares
      recovery_target_price = recovery_intrinsic_value * (1 - mos_percentage)
      recovery_valid = recovery_intrinsic_value > 0
      if recovery_valid and (base_fcf is None or base_fcf <= 0):
        intrinsic_value_dcf = recovery_intrinsic_value
        target_buy_price = recovery_target_price
        sum_pv_fcf = recovery_sum_pv_fcf
        pv_terminal_value = recovery_pv_terminal
        enterprise_value = recovery_ev
        equity_value = recovery_equity_value
        projected_fcfs = recovery_projected_fcfs
        pv_fcfs = recovery_pv_fcfs
        dcf_valid = True

  # --- RELATIVE VALUATION: proyeksi otomatis + target multiple yang dapat diedit ---
  eps = float(info.get("trailingEps") or 0)
  bvps = float(info.get("bookValue") or 0)
  sector = info.get("sector", "General")
  shares_for_rel = shares if shares and shares > 0 else 0
  revenue_per_share_reporting = (float(info.get("totalRevenue") or 0) / shares_for_rel) if shares_for_rel else 0
  sales_per_share_idr = revenue_per_share_reporting * financial_fx_multiplier
  ebitda_reporting = float(info.get("ebitda") or 0)
  ebitda_idr = ebitda_reporting * financial_fx_multiplier

  with st.sidebar.expander("📊 Asumsi Relative Valuation", expanded=False):
      st.caption(
          "Proyeksi EPS dan BVPS dihitung otomatis dari data terakhir dan asumsi pertumbuhan. "
          "Harga wajar dihitung setelah target PER/PBV diisi. Gunakan multiple historis atau peer yang relevan."
      )
      forward_eps_growth_pct = st.number_input(
          "Pertumbuhan EPS proyeksi (%)", min_value=-100.0, max_value=200.0,
          value=8.0, step=1.0,
          help="Digunakan untuk memproyeksikan EPS satu tahun ke depan bila data forward EPS dari sumber tidak tersedia."
      )
      forward_bvps_growth_pct = st.number_input(
          "Pertumbuhan BVPS proyeksi (%)", min_value=-100.0, max_value=100.0,
          value=5.0, step=1.0,
          help="Asumsi pertumbuhan nilai buku per saham satu tahun ke depan."
      )
      avg_per = st.number_input(
          "Target PER (x)", min_value=0.0, max_value=100.0, value=15.0, step=0.5,
          help="Acuan awal 15x. Bandingkan dengan PER rata-rata industri dan riwayat emiten."
      )
      avg_pbv = st.number_input(
          "Target PBV (x)", min_value=0.0, max_value=2.0, value=1.8, step=0.1,
          help="Acuan awal di bawah 2x. Bandingkan dengan PBV rata-rata industri dan riwayat emiten."
      )
      avg_psr = st.number_input(
          "Target PSR (x)", min_value=0.0, max_value=3.0, value=2.5, step=0.1,
          help="Acuan PSR di bawah 3x; sesuaikan dengan sektor dan margin perusahaan."
      )
      ev_ebitda_mult = st.number_input(
          "Target EV/EBITDA (x)", min_value=0.0, max_value=30.0, value=6.0, step=0.5,
          help="Acuan awal 6x untuk simulasi; bandingkan dengan multiple perusahaan sejenis."
      )
      st.caption(
          "Acuan awal, bukan angka universal. Wajib dibandingkan dengan PER/PBV rata-rata industri "
          "dan disesuaikan dengan sektor serta kondisi emiten."
      )

  st.sidebar.markdown("---")
  st.sidebar.subheader("💚 Dukung Pengembangan Dashboard")
  st.sidebar.caption(
      "Jika dashboard ini bermanfaat, dukungan sukarela membantu pengembangan fitur dan pemeliharaan Sintesainsights.com."
  )
  st.sidebar.markdown("[Donasi melalui Saweria](https://saweria.co/Limauwsn86)")


  # Forward EPS: prefer provider estimate; if unavailable, project from latest trailing EPS.
  forward_eps_source = info.get("forwardEps")
  forward_eps_provider = float(forward_eps_source) if forward_eps_source is not None else 0.0
  forward_eps = (
      forward_eps_provider if forward_eps_provider > 0
      else eps * (1 + forward_eps_growth_pct / 100) if eps > 0
      else 0.0
  )

  # Forward BVPS projection using the user-editable growth assumption.
  forward_bvps = bvps * (1 + forward_bvps_growth_pct / 100) if bvps > 0 else 0.0

  # Market forward multiples and intrinsic estimates are distinct concepts.
  market_forward_per = (float(current_price) / forward_eps) if forward_eps > 0 and current_price > 0 else None
  market_forward_pbv = (float(current_price) / forward_bvps) if forward_bvps > 0 and current_price > 0 else None
  val_per_price = forward_eps * avg_per if forward_eps > 0 and avg_per > 0 else None
  val_pbv_price = forward_bvps * avg_pbv if forward_bvps > 0 and avg_pbv > 0 else None

  # PSR and EV/EBITDA fair-value estimates, reported in IDR/share.
  val_psr_price = sales_per_share_idr * avg_psr if sales_per_share_idr > 0 and avg_psr > 0 else None
  val_ev_ebitda = (
      (ebitda_idr * ev_ebitda_mult + total_cash - total_debt) / shares
      if ebitda_idr > 0 and ev_ebitda_mult > 0 and shares > 0
      else None
  )
  relative_candidates = [
      p for p in [val_per_price, val_pbv_price, val_psr_price, val_ev_ebitda]
      if p is not None and np.isfinite(p) and p > 0
  ]
  avg_relative_value = float(np.mean(relative_candidates)) if relative_candidates else 0.0

  # --- DIVIDEND DISCOUNT MODEL (DDM) ---
  dividends = data["dividends"]
  ddm_value = 0
  has_consistent_dividend = False
  if not dividends.empty and len(dividends) >= 4:
    # Ambil total dividen 1 tahun terakhir
    recent_divs = dividends.tail(4).sum() * fx_multiplier
    if recent_divs > 0:
      has_consistent_dividend = True
      # Gordon Growth Model: DDM = D1 / (Ke - g)
      d1 = recent_divs * (1 + 0.02)  # Asumsi growth dividen 2%
      if cost_of_equity > 0.02:
        ddm_value = d1 / (cost_of_equity - 0.02)

  # --- TAMPILAN DASHBOARD UTAMA ---
  col1, col2, col3 = st.columns(3)
  col1.metric(
      "Harga Pasar Saat Ini",
      f"Rp {current_price:,.0f}",
      f"Kapitalisasi: {data['market_cap']/1e9:,.1f} Miliar",
  )
  col2.metric(
      "Nilai Intrinsik (DCF)",
      f"Rp {intrinsic_value_dcf:,.0f}" if dcf_valid else "Belum tersedia",
      f"MoS {int(mos_percentage*100)}%: Rp {target_buy_price:,.0f}" if dcf_valid else "FCF positif/WACC valid diperlukan",
  )
  col3.metric("Nilai Wajar (Relative)", f"Rp {avg_relative_value:,.0f}" if avg_relative_value > 0 else "Belum tersedia")

  st.markdown("---")

  # --- TAB VISUALISASI ---
  tab1, tab2, tab3 = st.tabs(
      [
          "📊 Ringkasan & Rekomendasi",
          "🏢 Relative & DDM Valuation",
          "📘 Panduan Valuasi",
      ]
  )

  with tab1:
    st.subheader(
        f"Kesimpulan Analisis Valuasi: {info.get('longName', ticker_full)}"
    )

    # Status valuasi: jangan keluarkan rekomendasi DCF jika basis FCF tidak valid.
    if not dcf_valid:
      st.warning(
          "⚠️ **DCF belum dapat dinilai.** FCF historis tidak tersedia/bernilai nol atau negatif, "
          "atau WACC tidak lebih besar dari pertumbuhan terminal. Periksa laporan arus kas dan asumsi. "
          "Dashboard tidak mengganti FCF dengan angka buatan."
      )
    elif current_price <= target_buy_price:
      st.success(
          f"🟢 **MENARIK (Undervalued)**: Harga pasar saat ini (Rp"
          f" {current_price:,.0f}) sudah berada di bawah Target Harga Beli dengan"
          f" Margin of Safety {int(mos_percentage*100)}% (Rp"
          f" {target_buy_price:,.0f})."
      )
    elif current_price <= intrinsic_value_dcf:
      st.warning(
          f"🟡 **CUKUP WAJAR (Waspada MoS)**: Harga pasar (Rp"
          f" {current_price:,.0f}) di bawah Nilai Intrinsik (Rp"
          f" {intrinsic_value_dcf:,.0f}), tetapi belum mencapai batas aman"
          f" Margin of Safety."
      )
    else:
      st.error(
          f"🔴 **OVERVALUED**: Harga pasar saat ini (Rp {current_price:,.0f})"
          f" sudah jauh melampaui Nilai Intrinsik DCF (Rp"
          f" {intrinsic_value_dcf:,.0f})."
      )

    col_a, col_b = st.columns(2)
    with col_a:
      st.markdown("### 📋 Jembatan Valuasi DCF")
      dcf_summary_df = pd.DataFrame({
          "Metrik": [
              f"Total PV Arus Kas ({projection_years} Thn)",
              "Present Value Terminal Value",
              "Enterprise Value (EV)",
              "Ditambah: Kas Total",
              "Dikurangi: Total Utang",
              "Equity Value",
              "Jumlah Saham Beredar",
          ],
          "Nilai (IDR)": [
              f"Rp {sum_pv_fcf:,.0f}" if dcf_valid else "Tidak dihitung",
              f"Rp {pv_terminal_value:,.0f}" if dcf_valid else "Tidak dihitung",
              f"Rp {enterprise_value:,.0f}" if dcf_valid else "Tidak dihitung",
              f"Rp {total_cash:,.0f}",
              f"Rp {total_debt:,.0f}",
              f"Rp {equity_value:,.0f}" if dcf_valid else "Tidak dihitung",
              f"{shares:,.0f}",
          ],
      })
      st.table(dcf_summary_df)

    with col_b:
      st.markdown("### 🎯 Perbandingan Harga Wajar")
      comparison_df = pd.DataFrame({
          "Metode Valuasi": [
              "Harga Pasar Aktual",
              f"DCF (MoS {int(mos_percentage*100)}%)",
              "DCF Intrinsic Value",
              "Relative Valuation (PER/PBV)",
              (
                  "Dividend Discount Model (DDM)"
                  if has_consistent_dividend
                  else "DDM (Tidak Konsisten/Kosong)"
              ),
          ],
          "Estimasi Harga": [
              f"Rp {current_price:,.0f}",
              f"Rp {target_buy_price:,.0f}" if dcf_valid else "DCF belum tersedia",
              f"Rp {intrinsic_value_dcf:,.0f}" if dcf_valid else "DCF belum tersedia",
              f"Rp {avg_relative_value:,.0f}",
              (
                  f"Rp {ddm_value:,.0f}"
                  if has_consistent_dividend
                  else "Kosong (-)"
              ),
          ],
      })
      st.table(comparison_df)

  with tab2:
    st.subheader("🏢 Pendekatan Relative Valuation & DDM")
    col_rel1, col_rel2 = st.columns([1.25, 1.0])

    with col_rel1:
      st.markdown("#### Relative Valuation Multiples")
      relative_rows = [
          ("Forward PER", val_per_price),
          ("Forward PBV", val_pbv_price),
          ("PSR", val_psr_price),
      ]
      # EV/EBITDA hanya ditampilkan bila EBITDA dan hasil per saham valid.
      if ebitda_idr > 0 and shares > 0 and val_ev_ebitda is not None and val_ev_ebitda > 0:
          relative_rows.append(("EV/EBITDA", val_ev_ebitda))
      st.dataframe(pd.DataFrame([
          {"Metode": name, "Estimasi Harga Wajar": f"Rp {value:,.0f}" if value is not None and value > 0 else "Belum tersedia"}
          for name, value in relative_rows
      ]), use_container_width=True, hide_index=True)




    with col_rel2:
      st.markdown("#### Dividend Discount Model (DDM)")
      if has_consistent_dividend:
        st.success(
            f"✅ **Emiten Membagikan Dividen Konsisten**\n\nEstimasi Harga Wajar"
            f" DDM: **Rp {ddm_value:,.0f}**"
        )
      else:
        st.warning(
            "⚠️ **Tidak Konsisten Membagikan Dividen**\n\nSesuai ketentuan,"
            " harga wajar DDM dikosongkan karena emiten tidak memiliki riwayat"
            " pembayaran dividen rutin yang memadai."
        )

  with tab3:
    st.header("📘 Panduan dan Penjelasan Metode Valuasi")
    with st.expander("1. Discounted Cash Flow (DCF)", expanded=False):
        st.markdown("""
        **Tujuan:** mengestimasi nilai perusahaan berdasarkan nilai sekarang dari arus kas bebas masa depan.

        - FCF historis dihitung/dibaca dari data arus kas; angka historis tidak diubah.
        - Proyeksi pemulihan memakai asumsi pertumbuhan pendapatan dan margin FCF yang bisa diedit di sidebar.
        - WACC adalah tingkat diskonto; pertumbuhan terminal harus lebih rendah dari WACC.
        - Nilai ekuitas = enterprise value + kas − total utang. Harga wajar per saham = nilai ekuitas / saham beredar.
        - Margin of Safety (MoS) adalah diskon terhadap estimasi nilai wajar, bukan jaminan keuntungan.

        **Peringatan:** hasil sensitif terhadap proyeksi, kurs, kas, utang, dan jumlah saham. Jika data penting tidak tersedia, hasil perlu dianggap indikatif atau tidak dihitung.
        """)
    with st.expander("2. Relative Valuation: PER, PBV, PSR, dan EV/EBITDA", expanded=False):
        st.markdown("""
        **Forward PER:** harga wajar = Forward EPS × target PER. Acuan awal 15x; bandingkan dengan rata-rata PER industri/sektor.

        **Forward PBV:** harga wajar = Forward BVPS × target PBV. Acuan awal 1,8x (di bawah 2x); bandingkan dengan rata-rata PBV industri/sektor.

        **PSR:** harga wajar = penjualan per saham × target PSR. Acuan awal 2,5x (di bawah 3x); pertimbangkan margin dan karakteristik sektor.

        **EV/EBITDA:** EV wajar = EBITDA × multiple target; harga wajar per saham = (EV wajar + kas − total utang) / saham beredar. Metode hanya ditampilkan bila data EBITDA valid.

        Multiple pembanding perlu berasal dari perusahaan sejenis dan periode data yang konsisten. Jika EPS, EBITDA, ekuitas, atau penjualan tidak valid/tersedia, metode terkait bisa tidak layak digunakan.
        """)
    with st.expander("3. Dividend Discount Model (DDM)", expanded=False):
        st.markdown("""
        DDM mengestimasi nilai saham dari dividen masa depan. Model Gordon Growth menggunakan rumus **D1 / (Ke − g)**, dengan D1 sebagai dividen per saham periode berikutnya, Ke sebagai biaya ekuitas, dan g sebagai pertumbuhan dividen jangka panjang. Rumus hanya layak bila Ke > g dan dividen diperkirakan berkelanjutan. Untuk emiten yang tidak rutin membagikan dividen, DDM bisa tidak relevan.
        """)
    with st.expander("4. Cara membaca hasil dan batasan data", expanded=False):
        st.markdown("""
        - **Harga pasar:** harga dari sumber data dan dapat terlambat/tidak selalu real-time.
        - **Nilai wajar:** estimasi berbasis asumsi, bukan harga pasti di masa depan.
        - **Multiple peer:** pengguna perlu memastikan perusahaan pembanding relevan.
        - **Mata uang:** laporan keuangan bisa berbeda mata uang dengan harga saham; periksa kurs dan label sumber.
        - **Data kosong/0:** tidak selalu berarti nilai aktual perusahaan nol; bisa berarti penyedia data tidak menyajikan akun tersebut.
        - Dashboard ini merupakan alat edukasi/analisis, bukan rekomendasi beli atau jual.
        """)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Indonesia Stock Valuation Dashboard (DCF, Relative, DDM)",
    page_icon="📈",
    layout="wide",
)

st.title("🇮🇩 IDR Stock Valuation Dashboard")
st.markdown(
    "Dashboard Valuasi Saham BEI Profesional (DCF CAPM, Relative Valuation, & DDM dengan Margin of Safety)."
)


def _latest_value_for_labels(balance_sheet, labels):
  """Ambil nilai dari akun yang cocok persis pada periode neraca terbaru."""
  if balance_sheet is None or balance_sheet.empty:
    return None, None, None

  # Kolom yfinance biasanya tanggal akhir periode; pilih tanggal terbaru secara eksplisit.
  columns = sorted(balance_sheet.columns, key=lambda x: pd.Timestamp(x), reverse=True)
  normalized_index = {str(idx).strip().lower(): idx for idx in balance_sheet.index}

  for label in labels:
    row = normalized_index.get(label.strip().lower())
    if row is None:
      continue
    for col in columns:
      value = balance_sheet.loc[row, col]
      if isinstance(value, pd.Series):
        value = value.iloc[0]
      if pd.notna(value):
        try:
          return float(value), str(col), str(row)
        except (TypeError, ValueError):
          continue
  return None, None, None


def _get_cash_and_debt(balance_sheet):
  """Ambil kas dan utang dari label neraca yang spesifik, tanpa mencampur periode."""
  cash, period, cash_label = _latest_value_for_labels(
      balance_sheet,
      ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"],
  )

  debt, debt_period, debt_label = _latest_value_for_labels(
      balance_sheet,
      ["Total Debt", "TotalDebt"],
  )

  # Jika total debt tidak tersedia, jumlahkan hanya akun pinjaman jangka pendek
  # dan panjang yang eksplisit. Jangan gunakan total liabilities sebagai utang.
  if debt is None:
    current_debt, current_period, current_label = _latest_value_for_labels(
        balance_sheet,
        ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Long Term Debt"],
    )
    long_debt, long_period, long_label = _latest_value_for_labels(
        balance_sheet,
        ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    )
    if current_debt is not None or long_debt is not None:
      debt = (current_debt or 0.0) + (long_debt or 0.0)
      debt_period = current_period or long_period
      debt_label = "Penjumlahan utang lancar + utang jangka panjang"

  # Kas dan utang wajib berasal dari tanggal laporan yang sama.
  # Bila akun ditemukan pada periode berbeda, tandai agar tidak digunakan diam-diam.
  if cash is not None and debt is not None and period and debt_period and period != debt_period:
    return cash, debt, {"cash_period": period, "debt_period": debt_period,
                        "cash_label": cash_label, "debt_label": debt_label,
                        "period_mismatch": True}
  return cash, debt, {"cash_period": period, "debt_period": debt_period,
                      "cash_label": cash_label, "debt_label": debt_label,
                      "period_mismatch": False}


@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol):
  try:
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    # Pisahkan mata uang harga saham dari mata uang laporan keuangan.
    # Untuk saham BEI, harga dapat dalam IDR sementara laporan keuangan emiten
    # (misalnya BUMI) disajikan dalam USD.
    currency = info.get("currency", "IDR")
    financial_currency = info.get("financialCurrency", currency)

    # Ambil data historis keuangan
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow

    # Harga terkini & Market Cap
    current_price = info.get(
        "currentPrice",
        info.get("regularMarketPrice", info.get("previousClose", 0)),
    )
    shares_outstanding = info.get("sharesOutstanding", 0)
    market_cap = info.get("marketCap", current_price * shares_outstanding)

    # Kas dan utang: gunakan label akun yang spesifik dan periode terbaru.
    total_cash, total_debt, balance_meta = _get_cash_and_debt(balance_sheet)
    if total_cash is None:
      total_cash = 0.0
    if total_debt is None:
      total_debt = 0.0

    # FCF historis: validasi angka Yahoo Finance terhadap arus kas operasi.
    # Jangan menerima FCF yang jauh di luar skala OCF; jika tidak valid, hitung
    # dari OCF + capex hanya bila kedua baris agregat tersedia dan cocok periodenya.
    fcf_list = []
    fcf_records = []
    fcf_source = "Tidak ditemukan / belum tervalidasi"
    if cashflow is not None and not cashflow.empty:
      cf_columns = sorted(cashflow.columns, key=lambda x: pd.Timestamp(x), reverse=True)

      def _norm_label(value):
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())

      normalized_cf = {_norm_label(idx): idx for idx in cashflow.index}
      direct_fcf_row = next(
        (normalized_cf[k] for k in ["freecashflow", "freecashflowcontinuing"] if k in normalized_cf),
        None,
      )
      ocf_row = next(
        (normalized_cf[k] for k in [
          "operatingcashflow",
          "cashflowfromcontinuingoperatingactivities",
          "totalcashfromoperatingactivities",
          "operatingcashflowcontinuing",
          "cashflowsfromusedinoperatingactivitiesdirect",
          "cashflowsfromusedinoperatingactivities",
          "cashflowsfromoperatingactivities",
        ] if k in normalized_cf),
        None,
      )
      # Yahoo Finance sometimes returns localized/expanded labels for the aggregate
      # operating cash flow. Use a narrow fallback only if exact aliases were absent.
      if ocf_row is None:
        for normalized_label, original_label in normalized_cf.items():
          if (
            "cashflowsfromusedinoperatingactivities" in normalized_label
            or "cashflowsfromoperatingactivities" in normalized_label
          ):
            ocf_row = original_label
            break
      capex_row = next(
        (normalized_cf[k] for k in [
          "capitalexpenditure",
          "capitalexpenditures",
          "capitalexpenditurereported",
          "capitalexpendituresreported",
          "purchaseofppe",
          "purchaseofpropertyplantequipment",
        ] if k in normalized_cf),
        None,
      )

      # Buat diagnostik per periode; simpan angka mentah yang membantu audit.
      for col in cf_columns:
        record = {"Periode": str(pd.Timestamp(col).date())}
        direct_value = cashflow.loc[direct_fcf_row, col] if direct_fcf_row is not None else np.nan
        ocf_value = cashflow.loc[ocf_row, col] if ocf_row is not None else np.nan
        capex_value = cashflow.loc[capex_row, col] if capex_row is not None else np.nan
        record["FCF Yahoo mentah (USD)"] = float(direct_value) if pd.notna(direct_value) else None
        record["OCF agregat (USD)"] = float(ocf_value) if pd.notna(ocf_value) else None
        record["CapEx (USD)"] = float(capex_value) if pd.notna(capex_value) else None

        accepted_value = None
        status = "Tidak dihitung: baris agregat OCF/CapEx tidak tersedia"
        if pd.notna(direct_value):
          direct_value = float(direct_value)
          # FCF puluhan/triliunan USD tidak wajar untuk emiten ini jika OCF hanya puluhan juta.
          if pd.notna(ocf_value):
            ocf_float = float(ocf_value)
            plausible = abs(direct_value) <= max(abs(ocf_float) * 10, 1_000_000_000)
          else:
            plausible = abs(direct_value) <= 10_000_000_000
          if plausible:
            accepted_value = direct_value
            status = "FCF Yahoo diterima; skala relatif terhadap OCF lolos pemeriksaan awal"
          elif pd.notna(ocf_value) and pd.notna(capex_value):
            calculated = float(ocf_value) + float(capex_value)
            # Tolak hasil turunan juga jika jauh dari skala OCF tanpa penjelasan.
            if abs(calculated) <= max(abs(float(ocf_value)) * 10, 1_000_000_000):
              accepted_value = calculated
              status = "FCF Yahoo ditolak sebagai outlier; memakai OCF + CapEx"
            else:
              status = "FCF Yahoo dan hasil OCF + CapEx tidak lolos pemeriksaan skala"
          else:
            status = "FCF Yahoo ditolak sebagai outlier; OCF/CapEx agregat tidak cukup untuk menghitung ulang"
        elif pd.notna(ocf_value) and pd.notna(capex_value):
          calculated = float(ocf_value) + float(capex_value)
          if abs(calculated) <= max(abs(float(ocf_value)) * 10, 1_000_000_000):
            accepted_value = calculated
            status = "Dihitung dari OCF + CapEx"
          else:
            status = "Hasil OCF + CapEx tidak lolos pemeriksaan skala"

        record["FCF tervalidasi (USD)"] = accepted_value
        record["Status pemeriksaan"] = status
        fcf_records.append(record)
        # DCF hanya memakai FCF dari periode paling baru; jangan diam-diam
        # mengganti tahun dasar dengan tahun lama jika periode terbaru invalid.
        if accepted_value is not None and col == cf_columns[0]:
          fcf_list.append(accepted_value)

      if direct_fcf_row is not None:
        fcf_source = f"Yahoo Finance: {direct_fcf_row} (diperiksa terhadap skala OCF)"
      elif ocf_row is not None and capex_row is not None:
        fcf_source = f"Dihitung dari {ocf_row} + {capex_row}"
    # Dividen history untuk DDM
    dividends = stock.dividends

    return {
        "info": info,
        "currency": currency,
        "financial_currency": financial_currency,
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "balance_meta": balance_meta,
        "fcf_list": fcf_list,
        "fcf_records": fcf_records,
        "fcf_source": fcf_source,
        "ocf_label_detected": str(ocf_row) if ocf_row is not None else None,
        "capex_label_detected": str(capex_row) if capex_row is not None else None,
        "dividends": dividends,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
    }, None
  except Exception as e:
    return None, str(e)


# Input Ticker di Sidebar
st.sidebar.header("🔍 Pencarian Emiten")
user_ticker = (
    st.sidebar.text_input("Masukkan Ticker IDX (Contoh: BBCA, ASII, ARCI)", "BBCA")
    .upper()
    .strip()
)
if not user_ticker.endswith(".JK") and user_ticker != "":
  ticker_full = user_ticker + ".JK"
else:
  ticker_full = user_ticker

data, err = get_stock_data(ticker_full)

# Ambil Kurs USDIDR jika emiten USD
usd_idr = 16000  # Fallback default
try:
  fx = yf.Ticker("USDIDR=X").info
  usd_idr = fx.get("regularMarketPrice", fx.get("previousClose", 16000))
except:
  pass

if err or not data or data["current_price"] == 0:
  st.error(
      f"Gagal memuat data untuk ticker {ticker_full}. Pastikan kode saham benar"
      " dan terdaftar di BEI."
  )
else:
  info = data["info"]
  currency = data["currency"]
  financial_currency = data.get("financial_currency", currency)
  # Multiplier umum untuk harga/per-share mengikuti mata uang quote.
  fx_multiplier = usd_idr if currency == "USD" else 1.0
  # Multiplier khusus data neraca dan arus kas mengikuti mata uang laporan.
  financial_fx_multiplier = usd_idr if financial_currency == "USD" else 1.0

  st.sidebar.success(
      f"Berhasil memuat: {info.get('longName', ticker_full)} ({currency})"
  )
  # Transparansi sumber akun neraca untuk membantu memvalidasi angka.
  balance_meta = data.get("balance_meta", {})
  if balance_meta.get("period_mismatch"):
    st.sidebar.warning(
        "Periode kas dan utang tidak sama. Periksa neraca sebelum menggunakan hasil valuasi. "
        f"Kas: {balance_meta.get('cash_period')}; utang: {balance_meta.get('debt_period')}"
    )
  with st.sidebar.expander("Detail sumber kas & utang"):
    st.write("Mata uang harga saham:", currency)
    st.write("Mata uang laporan keuangan:", financial_currency)
    st.write("Kurs USD/IDR yang digunakan:", f"Rp {usd_idr:,.2f}" if financial_currency == "USD" else "Tidak diperlukan")
    st.write("Akun kas:", balance_meta.get("cash_label") or "Tidak ditemukan")
    st.write("Periode kas:", balance_meta.get("cash_period") or "Tidak tersedia")
    st.write("Akun utang:", balance_meta.get("debt_label") or "Tidak ditemukan")
    st.write("Periode utang:", balance_meta.get("debt_period") or "Tidak tersedia")
    if not balance_meta.get("cash_label") or not balance_meta.get("debt_label"):
      st.warning("Salah satu akun tidak ditemukan di Yahoo Finance; nilai 0 bukan berarti saldo sebenarnya nol.")

  # Diagnostik sumber FCF: tampilkan akun arus kas mentah agar bisa memastikan
  # apakah FCF nol karena akun tidak dikenali atau memang nilai historisnya nol/negatif.
  with st.sidebar.expander("Diagnostik sumber FCF (arus kas)"):
    cashflow_df = data.get("cashflow")
    st.write("Mata uang laporan arus kas:", financial_currency)
    if cashflow_df is None or cashflow_df.empty:
      st.warning("Yahoo Finance tidak mengembalikan tabel arus kas untuk emiten ini.")
    else:
      st.write("Baris akun yang tersedia di Yahoo Finance:")
      st.dataframe(cashflow_df, use_container_width=True)
      st.write("Nama akun lengkap (untuk pemeriksaan pencocokan):")
      st.code("\\n".join(
        f"ASLI: {repr(idx)} | NORMAL: {''.join(ch.lower() for ch in str(idx) if ch.isalnum())}"
        for idx in cashflow_df.index
      ))
      st.write("Baris OCF yang berhasil dicocokkan:", data.get("ocf_label_detected") or "TIDAK DITEMUKAN")
      st.write("Baris CapEx yang berhasil dicocokkan:", data.get("capex_label_detected") or "TIDAK DITEMUKAN")
      st.write("Sumber FCF yang digunakan:", data.get("fcf_source", "Tidak diketahui"))
      st.write("Pemeriksaan FCF per periode (sebelum konversi kurs):")
      fcf_records = data.get("fcf_records", [])
      if fcf_records:
        st.dataframe(pd.DataFrame(fcf_records), use_container_width=True, hide_index=True)
      else:
        st.warning("Tidak ada nilai FCF historis yang berhasil dibaca.")
      st.caption("FCF mentah yang tidak konsisten dengan skala OCF ditolak; FCF tervalidasi saja yang dapat masuk ke perhitungan DCF.")

  # --- SIDEBAR: ASUMSI VALUASI ---
  st.sidebar.markdown("---")
  st.sidebar.header("⚙️ Pengaturan Asumsi Valuasi")

  # 1. Parameter WACC (CAPM Components)
  st.sidebar.subheader("A. Komponen WACC (CAPM)")
  risk_free_rate = (
      st.sidebar.number_input(
          "Risk-Free Rate Rf (%)", min_value=0.0, max_value=20.0, value=6.5, step=0.1
      )
      / 100
  )
  tax_rate = (
      st.sidebar.number_input(
          "Tax Rate Perusahaan (%)",
          min_value=0.0,
          max_value=50.0,
          value=22.0,
          step=0.5,
      )
      / 100
  )
  market_risk_premium = (
      st.sidebar.number_input(
          "Market Risk Premium (%)",
          min_value=0.0,
          max_value=20.0,
          value=6.0,
          step=0.1,
      )
      / 100
  )
  beta = st.sidebar.number_input(
      "Beta Saham", min_value=0.0, max_value=3.0, value=1.0, step=0.05
  )
  cost_of_debt = (
      st.sidebar.number_input(
          "Cost of Debt Sebelum Pajak (%)",
          min_value=0.0,
          max_value=30.0,
          value=8.0,
          step=0.5,
      )
      / 100
  )
  debt_weight = (
      st.sidebar.slider(
          "Bobot Utang / Capital Structure Debt (%)", 0, 100, 20, 5
      )
      / 100
  )
  equity_weight = 1.0 - debt_weight

  # Hitung WACC otomatis
  cost_of_equity = risk_free_rate + (beta * market_risk_premium)
  effective_cost_of_debt = cost_of_debt * (1 - tax_rate)
  calculated_wacc = (equity_weight * cost_of_equity) + (
      debt_weight * effective_cost_of_debt
  )

  st.sidebar.info(
      f"**Cost of Equity (Ke):** {cost_of_equity*100:.2f}%\n\n"
      f"**Cost of Debt (After Tax):** {effective_cost_of_debt*100:.2f}%\n\n"
      f"⚡ **Total WACC Otomatis:** **{calculated_wacc*100:.2f}%**"
  )

  # 2. Parameter DCF Lainnya
  st.sidebar.subheader("B. Parameter Proyeksi DCF")
  projection_years = st.sidebar.selectbox(
      "Time Period Cash Flow", options=[5, 10], index=0
  )
  fcf_growth_rate = (
      st.sidebar.slider(
          "Pertumbuhan FCF Tahunan (%)", -10.0, 30.0, 8.0, 0.5
      )
      / 100
  )
  perpetual_growth = (
      st.sidebar.slider("Perpetual Growth Rate (%)", 0.0, 5.0, 2.5, 0.25)
      / 100
  )

  # 3. Margin of Safety
  st.sidebar.subheader("C. Margin of Safety (MoS)")
  mos_percentage = (
      st.sidebar.selectbox(
          "Pilih Margin of Safety",
          options=[10, 15, 20, 25, 30, 35, 40, 45, 50],
          index=3,
      )
      / 100
  )

  # --- MAIN DATA PROCESSING ---
  current_price = data["current_price"]
  shares = data["shares_outstanding"]
  total_debt = data["total_debt"] * financial_fx_multiplier
  total_cash = data["total_cash"] * financial_fx_multiplier
  fcf_list = data["fcf_list"]

  # DCF hanya dihitung jika FCF historis tersedia dan positif.
  # Jangan mengganti FCF negatif/kosong dengan angka buatan karena dapat menyesatkan valuasi.
  base_fcf = fcf_list[0] * financial_fx_multiplier if len(fcf_list) > 0 else None
  dcf_valid = (
      base_fcf is not None
      and base_fcf > 0
      and calculated_wacc > perpetual_growth
      and shares > 0
  )

  projected_fcfs = []
  pv_fcfs = []
  if dcf_valid:
    current_fcf = base_fcf
    for yr in range(1, projection_years + 1):
      current_fcf *= 1 + fcf_growth_rate
      projected_fcfs.append(current_fcf)
      pv_fcfs.append(current_fcf / ((1 + calculated_wacc) ** yr))
    sum_pv_fcf = sum(pv_fcfs)
    terminal_value = (projected_fcfs[-1] * (1 + perpetual_growth)) / (calculated_wacc - perpetual_growth)
    pv_terminal_value = terminal_value / ((1 + calculated_wacc) ** projection_years)
    enterprise_value = sum_pv_fcf + pv_terminal_value
    equity_value = enterprise_value + total_cash - total_debt
    intrinsic_value_dcf = equity_value / shares
    target_buy_price = intrinsic_value_dcf * (1 - mos_percentage)
  else:
    sum_pv_fcf = pv_terminal_value = enterprise_value = equity_value = None
    intrinsic_value_dcf = target_buy_price = None
    projected_fcfs = [None] * projection_years
    pv_fcfs = [None] * projection_years

  # --- RELATIVE VALUATION ---
  # EPS dan bookValue dari Yahoo Finance adalah metrik per saham dalam mata uang kuotasi saham.
  # Untuk saham BEI, nilainya sudah dalam IDR/share; jangan dikalikan kurs laporan keuangan USD.
  eps = info.get("trailingEps", info.get("forwardEps", 0)) or 0
  bvps = info.get("bookValue", 0) or 0
  sector = info.get("sector", "General")

  # Asumsi rata-rata industri standar BEI untuk pembanding
  avg_per = 15.0
  avg_pbv = 2.0
  val_per_price = eps * avg_per if eps > 0 else 0
  val_pbv_price = bvps * avg_pbv if bvps > 0 else 0

  # EV/EBITDA dan EV/Sales Approximation
  ebitda = info.get("ebitda", 0) * financial_fx_multiplier
  total_revenue = info.get("totalRevenue", 0) * financial_fx_multiplier
  ev_ebitda_mult = 10.0
  val_ev_ebitda = (
      ((ebitda * ev_ebitda_mult) + total_cash - total_debt) / shares
      if ebitda > 0 and shares > 0
      else 0
  )

  # Nilai Relatif Gabungan
  relative_prices = [p for p in [val_per_price, val_pbv_price] if p > 0]
  avg_relative_value = (
      sum(relative_prices) / len(relative_prices) if relative_prices else 0
  )

  # --- DIVIDEND DISCOUNT MODEL (DDM) ---
  dividends = data["dividends"]
  ddm_value = 0
  has_consistent_dividend = False
  if not dividends.empty and len(dividends) >= 4:
    # Ambil total dividen 1 tahun terakhir
    recent_divs = dividends.tail(4).sum() * fx_multiplier
    if recent_divs > 0:
      has_consistent_dividend = True
      # Gordon Growth Model: DDM = D1 / (Ke - g)
      d1 = recent_divs * (1 + 0.02)  # Asumsi growth dividen 2%
      if cost_of_equity > 0.02:
        ddm_value = d1 / (cost_of_equity - 0.02)

  # --- TAMPILAN DASHBOARD UTAMA ---
  col1, col2, col3 = st.columns(3)
  col1.metric(
      "Harga Pasar Saat Ini",
      f"Rp {current_price:,.0f}",
      f"Kapitalisasi: {data['market_cap']/1e9:,.1f} Miliar",
  )
  col2.metric(
      "Nilai Intrinsik (DCF)",
      f"Rp {intrinsic_value_dcf:,.0f}" if dcf_valid else "Belum tersedia",
      f"MoS {int(mos_percentage*100)}%: Rp {target_buy_price:,.0f}" if dcf_valid else "FCF positif/WACC valid diperlukan",
  )
  col3.metric("Nilai Wajar (Relative)", f"Rp {avg_relative_value:,.0f}")

  st.markdown("---")

  # --- TAB VISUALISASI ---
  tab1, tab2, tab3 = st.tabs(
      [
          "📊 Ringkasan & Rekomendasi",
          "📈 Proyeksi DCF Detail",
          "🏢 Relative & DDM Valuation",
      ]
  )

  with tab1:
    st.subheader(
        f"Kesimpulan Analisis Valuasi: {info.get('longName', ticker_full)}"
    )

    # Status valuasi: jangan keluarkan rekomendasi DCF jika basis FCF tidak valid.
    if not dcf_valid:
      st.warning(
          "⚠️ **DCF belum dapat dinilai.** FCF historis tidak tersedia/bernilai nol atau negatif, "
          "atau WACC tidak lebih besar dari pertumbuhan terminal. Periksa laporan arus kas dan asumsi. "
          "Dashboard tidak mengganti FCF dengan angka buatan."
      )
    elif current_price <= target_buy_price:
      st.success(
          f"🟢 **MENARIK (Undervalued)**: Harga pasar saat ini (Rp"
          f" {current_price:,.0f}) sudah berada di bawah Target Harga Beli dengan"
          f" Margin of Safety {int(mos_percentage*100)}% (Rp"
          f" {target_buy_price:,.0f})."
      )
    elif current_price <= intrinsic_value_dcf:
      st.warning(
          f"🟡 **CUKUP WAJAR (Waspada MoS)**: Harga pasar (Rp"
          f" {current_price:,.0f}) di bawah Nilai Intrinsik (Rp"
          f" {intrinsic_value_dcf:,.0f}), tetapi belum mencapai batas aman"
          f" Margin of Safety."
      )
    else:
      st.error(
          f"🔴 **OVERVALUED**: Harga pasar saat ini (Rp {current_price:,.0f})"
          f" sudah jauh melampaui Nilai Intrinsik DCF (Rp"
          f" {intrinsic_value_dcf:,.0f})."
      )

    col_a, col_b = st.columns(2)
    with col_a:
      st.markdown("### 📋 Jembatan Valuasi DCF")
      dcf_summary_df = pd.DataFrame({
          "Metrik": [
              f"Total PV Arus Kas ({projection_years} Thn)",
              "Present Value Terminal Value",
              "Enterprise Value (EV)",
              "Ditambah: Kas Total",
              "Dikurangi: Total Utang",
              "Equity Value",
              "Jumlah Saham Beredar",
          ],
          "Nilai (IDR)": [
              f"Rp {sum_pv_fcf:,.0f}" if dcf_valid else "Tidak dihitung",
              f"Rp {pv_terminal_value:,.0f}" if dcf_valid else "Tidak dihitung",
              f"Rp {enterprise_value:,.0f}" if dcf_valid else "Tidak dihitung",
              f"Rp {total_cash:,.0f}",
              f"Rp {total_debt:,.0f}",
              f"Rp {equity_value:,.0f}" if dcf_valid else "Tidak dihitung",
              f"{shares:,.0f}",
          ],
      })
      st.table(dcf_summary_df)

    with col_b:
      st.markdown("### 🎯 Perbandingan Harga Wajar")
      comparison_df = pd.DataFrame({
          "Metode Valuasi": [
              "Harga Pasar Aktual",
              f"DCF (MoS {int(mos_percentage*100)}%)",
              "DCF Intrinsic Value",
              "Relative Valuation (PER/PBV)",
              (
                  "Dividend Discount Model (DDM)"
                  if has_consistent_dividend
                  else "DDM (Tidak Konsisten/Kosong)"
              ),
          ],
          "Estimasi Harga": [
              f"Rp {current_price:,.0f}",
              f"Rp {target_buy_price:,.0f}" if dcf_valid else "DCF belum tersedia",
              f"Rp {intrinsic_value_dcf:,.0f}" if dcf_valid else "DCF belum tersedia",
              f"Rp {avg_relative_value:,.0f}",
              (
                  f"Rp {ddm_value:,.0f}"
                  if has_consistent_dividend
                  else "Kosong (-)"
              ),
          ],
      })
      st.table(comparison_df)

  with tab2:
    st.subheader(f"Proyeksian Arus Kas & Terminal Value ({projection_years} Tahun)")
    proj_years_list = [f"Tahun {i}" for i in range(1, projection_years + 1)]
    proj_df = pd.DataFrame({
        "Periode": proj_years_list,
        "Proyeksi FCF (IDR)": projected_fcfs,
        "Present Value FCF (IDR)": pv_fcfs,
    })
    st.dataframe(proj_df, use_container_width=True)

    # Grafik Visualisasi Proyeksian FCF
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=proj_years_list,
            y=projected_fcfs,
            name="Proyeksi FCF Nominal",
            marker_color="indianred",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=proj_years_list,
            y=pv_fcfs,
            name="Present Value FCF",
            mode="lines+markers",
            marker_color="royalblue",
        )
    )
    fig.update_layout(
        title="Grafik Proyeksi Arus Kas Bebas (FCF)",
        xaxis_title="Periode Proyeksi",
        yaxis_title="Nilai (IDR)",
    )
    st.plotly_chart(fig, use_container_width=True)

  with tab3:
    st.subheader("🏢 Pendekatan Relative Valuation & DDM")
    col_rel1, col_rel2 = st.columns(2)

    with col_rel1:
      st.markdown("#### Relative Valuation Multiples")
      st.write(
          f"• **Valuasi Berbasis PER (Asumsi PER 15x):** Rp"
          f" {val_per_price:,.0f}"
      )
      st.write(
          f"• **Valuasi Berbasis PBV (Asumsi PBV 2x):** Rp"
          f" {val_pbv_price:,.0f}"
      )
      st.write(
          f"• **Valuasi Berbasis EV/EBITDA:** Rp {val_ev_ebitda:,.0f}"
      )
      st.info(
          "Relative valuation membandingkan kelipatan harga emiten terhadap"
          " laba bersih (PER) dan nilai buku (PBV) rata-rata industri."
      )

    with col_rel2:
      st.markdown("#### Dividend Discount Model (DDM)")
      if has_consistent_dividend:
        st.success(
            f"✅ **Emiten Membagikan Dividen Konsisten**\n\nEstimasi Harga Wajar"
            f" DDM: **Rp {ddm_value:,.0f}**"
        )
      else:
        st.warning(
            "⚠️ **Tidak Konsisten Membagikan Dividen**\n\nSesuai ketentuan,"
            " harga wajar DDM dikosongkan karena emiten tidak memiliki riwayat"
            " pembayaran dividen rutin yang memadai."
        )
