const form = document.querySelector('#analysis-form');
const button = form.querySelector('button');
const status = document.querySelector('#status');
const error = document.querySelector('#error');
const result = document.querySelector('#result');
const imageInput = form.elements.image;
const imagePreview = document.querySelector('#image-preview');
const imagePreviewImage = document.querySelector('#image-preview-image');
let previewUrl = null;
let volumeZoneChart = null;
let featureRevision = 0;
const chartContainer = document.querySelector('#volume-zone-chart-container');

form.querySelectorAll('[name="additional_feature"]').forEach(input => input.addEventListener('change', () => {
  featureRevision++;
  result.hidden = true;
  destroyVolumeZoneChart();
}));

function destroyVolumeZoneChart() {
  if (volumeZoneChart) {
    volumeZoneChart.destroy();
    volumeZoneChart = null;
  }
  chartContainer.hidden = true;
}

function showVolumeZoneChart(data) {
  const points = data.volume_zone_distribution.points.map(point => ({
    x: point.distance, y: point.relative_density
  }));
  const mode = data.volume_zone_distribution.mode_distance;
  chartContainer.hidden = false;
  volumeZoneChart = new Chart(document.querySelector('#volume-zone-chart'), {
    type: 'line',
    data: { datasets: [
      { label: '距離分布', data: points, borderColor: '#245d85', pointRadius: 0, borderWidth: 2 },
      { label: 'この顔', data: [{ x: data.distance, y: 0 }, { x: data.distance, y: data.volume_zone_score }],
        borderColor: '#bd6038', backgroundColor: '#bd6038', pointRadius: [0, 5], borderWidth: 2 },
      { label: '最頻距離', data: [{ x: mode, y: 0 }, { x: mode, y: 100 }],
        borderColor: '#687985', borderDash: [5, 5], pointRadius: 0, borderWidth: 2 }
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { type: 'linear', min: points[0].x, max: points[points.length - 1].x,
          title: { display: true, text: '平均顔からの距離 D' } },
        y: { min: 0, max: 100, title: { display: true, text: '相対的な現れやすさ' } }
      }
    }
  });
}

imageInput.addEventListener('change', () => {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  const file = imageInput.files[0];
  if (!file) {
    imagePreviewImage.removeAttribute('src');
    imagePreview.hidden = true;
    return;
  }

  previewUrl = URL.createObjectURL(file);
  imagePreviewImage.src = previewUrl;
  imagePreview.hidden = false;
});

window.addEventListener('pagehide', () => {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
});

const messages = {
  no_face: '顔が検出されませんでした。正面を向いた顔が写る画像を選んでください。',
  multiple_faces: '複数の顔が検出されました。1人だけが写る画像を選んでください。',
  validation_failed: '画像が解析条件を満たしていません。明るく鮮明な正面顔の画像を選んでください。',
  invalid_image: '画像形式が不正です。JPEGまたはPNG画像を選んでください。',
  image_size_limit: '画像のファイルサイズまたは寸法が上限を超えています。小さい画像を選んでください。',
  model_unavailable: '解析モデルを利用できません。しばらくしてからお試しください。',
  invalid_reference_group: '参照集団を選択してください。',
  invalid_feature_selection: '追加項目の選択が不正です。選択内容を確認してください。',
  invalid_request: '入力内容を確認してください。',
  analysis_error: '解析できませんでした。別の画像でお試しください。'
};

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (button.disabled) return;
  button.disabled = true;
  result.hidden = true;
  destroyVolumeZoneChart();
  error.hidden = true;
  status.textContent = '解析中...';
  const groupLabel = form.elements.reference_group.selectedOptions[0].textContent;
  const submittedRevision = featureRevision;
  try {
    const response = await fetch('/analyze', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (submittedRevision !== featureRevision) return;
    if (!response.ok) {
      const issues = data.error?.issues;
      error.textContent = Array.isArray(issues) && issues.length
        ? (issues.length === 1 ? issues[0].message : `次の問題があります：\n${issues.map((issue) => `・${issue.message}`).join('\n')}`)
        : (messages[data.error?.code] || messages.analysis_error);
      error.hidden = false;
      return;
    }
    document.querySelector('#score').textContent = String(data.score);
    document.querySelector('#distance').textContent = String(data.distance);
    document.querySelector('#volume-zone-score').textContent = String(data.volume_zone_score);
    document.querySelector('#result-group').textContent = groupLabel;
    document.querySelector('#result-profile').textContent = data.profile_id === 'core3+face_aspect_ratio'
      ? '基本3特徴 + 顔の縦横比（実験的）' : '基本3特徴';
    document.querySelector('#result-dimension').textContent = String(data.dimension);
    const aspectDetail = document.querySelector('#aspect-detail');
    aspectDetail.hidden = !data.used_features.includes('face_aspect_ratio');
    if (!aspectDetail.hidden) {
      document.querySelector('#aspect-value').textContent = data.features.face_aspect_ratio.toFixed(4);
      document.querySelector('#aspect-z').textContent = data.z_scores.face_aspect_ratio.toFixed(4);
    }
    document.querySelector('#notice').textContent = data.approximation_notice;
    document.querySelector('#quality-warnings').textContent = [...new Set((data.quality_warnings || []).map(warning => warning.message))].join('\n');
    result.hidden = false;
    try {
      showVolumeZoneChart(data);
    } catch (_) {
      // 数値結果はグラフの描画失敗にかかわらず表示する。
      chartContainer.hidden = true;
      try {
        const existingChart = typeof Chart !== 'undefined' && Chart.getChart('volume-zone-chart');
        if (existingChart) existingChart.destroy();
      } catch (_) {
        // 描画失敗時の後片付けも数値結果の表示を妨げない。
      }
      volumeZoneChart = null;
    }
  } catch (_) {
    error.textContent = '通信または解析に失敗しました。時間をおいて再度お試しください。';
    error.hidden = false;
  } finally {
    status.textContent = '';
    button.disabled = false;
  }
});
