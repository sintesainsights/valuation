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