const form = document.querySelector('#analysis-form');
const button = form.querySelector('button');
const status = document.querySelector('#status');
const error = document.querySelector('#error');
const result = document.querySelector('#result');
const imageInput = form.elements.image;
const imagePreview = document.querySelector('#image-preview');
const imagePreviewImage = document.querySelector('#image-preview-image');
let previewUrl = null;

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
  invalid_request: '入力内容を確認してください。',
  analysis_error: '解析できませんでした。別の画像でお試しください。'
};

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (button.disabled) return;
  button.disabled = true;
  result.hidden = true;
  error.hidden = true;
  status.textContent = '解析中...';
  const groupLabel = form.elements.reference_group.selectedOptions[0].textContent;
  try {
    const response = await fetch('/analyze', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
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
    document.querySelector('#notice').textContent = data.approximation_notice;
    document.querySelector('#quality-warnings').textContent = [...new Set((data.quality_warnings || []).map(warning => warning.message))].join('\n');
    result.hidden = false;
  } catch (_) {
    error.textContent = '通信または解析に失敗しました。時間をおいて再度お試しください。';
    error.hidden = false;
  } finally {
    status.textContent = '';
    button.disabled = false;
  }
});
