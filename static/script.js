// static/script.js
let spendRevenueChart = null;
let ctrChart = null;

function initCharts() {
  const ctx = document.getElementById("spendRevenueChart").getContext("2d");
  spendRevenueChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [], datasets: [
        { label: "Spend", data: [], borderWidth: 2, tension:0.3, fill:false },
        { label: "Revenue", data: [], borderWidth: 2, tension:0.3, fill:false }
      ]
    },
    options: { responsive:true, plugins:{legend:{position:'top'}} }
  });

  const ctx2 = document.getElementById("ctrChart").getContext("2d");
  ctrChart = new Chart(ctx2, {
    type: "bar",
    data: { labels: [], datasets: [{ label: "CTR", data: [], borderWidth:1 }] },
    options: { responsive:true }
  });
}

function refreshDatasetSummary(){
  fetch("/dashboard")
    .then(r => r.text())
    .then(html => {
      // We request stats via /dashboard (rendered), but easier: call /dashboard to parse initial stats embedded?
      // Instead request /api/summary (not implemented) — fallback: load stats with an endpoint not present.
      // For robustness, call index.html to just update values using fetch to get ad_data.csv
      fetch("/static/").catch(()=>{});
    });
  // Simpler: load ad_data.csv if exists
  fetch("/ad_data.csv").then(r=>{
    if(!r.ok) return;
    return r.text();
  }).then(txt=>{
    if(!txt) return;
    const lines = txt.split("\n").filter(l=>l.trim().length>0);
    const header = lines[0].split(",");
    const rows = lines.slice(1).map(l=>l.split(","));
    const colIndex = (name) => header.indexOf(name);
    const ctrIdx = colIndex("ctr");
    const cvrIdx = colIndex("cvr");
    const costIdx = colIndex("cost");
    const revenueIdx = colIndex("revenue");
    const rowsNum = rows.length;
    let sumCtr=0,sumCvr=0,sumCost=0,sumRev=0;
    rows.forEach(r=>{
      if(ctrIdx>=0) sumCtr += parseFloat(r[ctrIdx]||0);
      if(cvrIdx>=0) sumCvr += parseFloat(r[cvrIdx]||0);
      if(costIdx>=0) sumCost += parseFloat(r[costIdx]||0);
      if(revenueIdx>=0) sumRev += parseFloat(r[revenueIdx]||0);
    });
    document.getElementById("statRows").innerText = rowsNum;
    document.getElementById("statCtr").innerText = (sumCtr/Math.max(1,rowsNum)).toFixed(4);
    document.getElementById("statCvr").innerText = (sumCvr/Math.max(1,rowsNum)).toFixed(4);
    document.getElementById("statSpend").innerText = sumCost.toFixed(2);
    document.getElementById("statRevenue").innerText = sumRev.toFixed(2);
    // update top charts
    const windowLabels = [ "t1","t2","t3","t4","t5","t6","t7","t8" ];
    const spendSeries = Array(windowLabels.length).fill(sumCost/windowLabels.length);
    const revenueSeries = Array(windowLabels.length).fill(sumRev/windowLabels.length);
    spendRevenueChart.data.labels = windowLabels;
    spendRevenueChart.data.datasets[0].data = spendSeries;
    spendRevenueChart.data.datasets[1].data = revenueSeries;
    spendRevenueChart.update();
    ctrChart.data.labels = windowLabels;
    ctrChart.data.datasets[0].data = Array(windowLabels.length).fill((sumCtr/Math.max(1,rowsNum)));
    ctrChart.update();
  }).catch(()=>{});
}

document.addEventListener("DOMContentLoaded", function(){
  initCharts();
  refreshDatasetSummary();
  // Buttons
  document.getElementById("btnTrain").onclick = function(){
    const timesteps = parseInt(document.getElementById("timesteps").value||20000);
    const sampleRows = parseInt(document.getElementById("sampleRows").value||50000);
    fetch("/api/train", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({timesteps:timesteps, sample_rows:sampleRows})})
      .then(r=>r.json()).then(j=>{ document.getElementById("trainStatus").innerText = j.message || JSON.stringify(j); pollTrainStatus(); });
  };

  function pollTrainStatus(){
    fetch("/api/train-status").then(r=>r.json()).then(s=>{
      document.getElementById("trainStatus").innerText = s.message || JSON.stringify(s);
      if(s.running) setTimeout(pollTrainStatus, 1500);
      else refreshDatasetSummary();
    });
  }

  document.getElementById("btnEval").onclick = function(){
    fetch("/api/evaluate", {method:"POST"}).then(r=>r.json()).then(j=>{ document.getElementById("evalStatus").innerText = j.message || JSON.stringify(j); pollEvalStatus(); });
  };

  function pollEvalStatus(){
    fetch("/api/eval-status").then(r=>r.json()).then(s=>{
      document.getElementById("evalStatus").innerText = s.message || JSON.stringify(s);
      if(s.running) setTimeout(pollEvalStatus, 1500);
      else {
        if(s.roas !== undefined && s.roas !== null){
          document.getElementById("achievedValue").innerText = s.roas.toFixed(2);
          document.getElementById("spendValue").innerText = s.spend.toFixed(2);
          document.getElementById("revenueValue").innerText = s.revenue.toFixed(2);
        }
      }
    });
  }

  document.getElementById("btnUpload").onclick = function(){
    const f = document.getElementById("fileInput").files[0];
    if(!f){ alert("Choose a CSV file."); return; }
    const fd = new FormData(); fd.append("file", f);
    fetch("/api/upload", {method:"POST", body: fd}).then(r=>r.json()).then(j=>{ alert(j.message); refreshDatasetSummary(); });
  };

  // initial poll to fill statuses
  setTimeout(()=>{ pollTrainStatus(); pollEvalStatus(); }, 500);
});
