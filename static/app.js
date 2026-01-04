document.addEventListener('DOMContentLoaded', () => {
    // State
    let selectedImagePath = null;

    // Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    // Stats
    const papersCountEl = document.getElementById('papers-count');
    const imagesCountEl = document.getElementById('images-count');

    // Papers
    const paperInput = document.getElementById('paper-input');
    const paperDropZone = document.getElementById('paper-drop-zone');
    const uploadPaperBtn = document.getElementById('upload-paper-btn');
    const paperTopicsInput = document.getElementById('paper-topics');
    const useLlmCheckbox = document.getElementById('use-llm');
    const searchPaperBtn = document.getElementById('search-paper-btn');
    const paperQueryInput = document.getElementById('paper-query');
    const paperResultsContainer = document.getElementById('paper-results');

    // Images
    const imageInput = document.getElementById('image-input');
    const imageDropZone = document.getElementById('image-drop-zone');
    const uploadImageBtn = document.getElementById('upload-image-btn');
    const searchImageBtn = document.getElementById('search-image-btn');
    const imageQueryInput = document.getElementById('image-query');
    const imageResultsContainer = document.getElementById('image-results');

    // Vision
    const visionPreview = document.getElementById('vision-preview');
    const describeBtn = document.getElementById('describe-btn');
    const askBtn = document.getElementById('ask-btn');
    const visualQuestionInput = document.getElementById('visual-question');
    const visionResultText = document.getElementById('vision-result-text');

    // Batch Organization
    const unorganizedList = document.getElementById('unorganized-list');
    const refreshUnorganizedBtn = document.getElementById('refresh-unorganized-btn');
    const batchOrganizeBtn = document.getElementById('batch-organize-btn');

    // --- Tab Switching ---
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');

            if (tabId === 'papers') refreshUnorganized();
        });
    });

    // --- Utils ---
    const showLoading = (text = "处理中...") => {
        loadingText.textContent = text;
        loadingOverlay.classList.remove('hidden');
    };

    const hideLoading = () => {
        loadingOverlay.classList.add('hidden');
    };

    const updateStats = async () => {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            papersCountEl.textContent = `Documents: ${data.papers_indexed}`;
            imagesCountEl.textContent = `Images: ${data.images_indexed}`;
        } catch (e) {
            console.error("Failed to fetch stats", e);
        }
    };

    const refreshUnorganized = async () => {
        try {
            const res = await fetch('/api/papers/unorganized');
            const data = await res.json();

            unorganizedList.innerHTML = '';
            if (data.length === 0) {
                unorganizedList.innerHTML = '<div class="placeholder">暂无待整理的论文</div>';
            } else {
                data.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'unorganized-item';
                    div.innerHTML = `<i class="fas fa-file-pdf"></i><span>${item.filename}</span>`;
                    unorganizedList.appendChild(div);
                });
            }
        } catch (e) {
            console.error("Failed to fetch unorganized papers", e);
        }
    };

    // --- Paper Management ---
    paperDropZone.addEventListener('click', () => paperInput.click());
    paperDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        paperDropZone.classList.add('dragover');
    });
    paperDropZone.addEventListener('dragleave', () => paperDropZone.classList.remove('dragover'));
    paperDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        paperDropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            paperInput.files = e.dataTransfer.files;
            paperDropZone.querySelector('p').textContent = e.dataTransfer.files[0].name;
        }
    });

    paperInput.addEventListener('change', () => {
        if (paperInput.files.length > 0) {
            paperDropZone.querySelector('p').textContent = paperInput.files[0].name;
        }
    });

    uploadPaperBtn.addEventListener('click', async () => {
        if (paperInput.files.length === 0) return alert("请先选择 PDF 文件");

        showLoading("正在上传并处理 PDF...");
        const formData = new FormData();
        formData.append('file', paperInput.files[0]);
        formData.append('topics', paperTopicsInput.value);
        formData.append('use_llm', useLlmCheckbox.checked);

        try {
            const res = await fetch('/api/papers/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                alert(`上传成功！归类为: ${data.topic}`);
                updateStats();
                refreshUnorganized();
            } else {
                alert(`出错: ${data.detail}`);
            }
        } catch (e) {
            alert("上传失败，请检查后端服务");
        } finally {
            hideLoading();
        }
    });

    refreshUnorganizedBtn.addEventListener('click', refreshUnorganized);

    batchOrganizeBtn.addEventListener('click', async () => {
        const topics = paperTopicsInput.value.trim();
        if (!topics) return alert("请先在上方输入分类主题（如：CV, NLP）");

        showLoading("正在批量执行整理过程...");
        try {
            const res = await fetch('/api/papers/batch_organize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    topics,
                    use_llm: useLlmCheckbox.checked
                })
            });
            const results = await res.json();

            const successCount = results.filter(r => r.status === 'success').length;
            const failCount = results.length - successCount;

            alert(`整理完成！成功: ${successCount} 篇, 失败: ${failCount} 篇`);
            updateStats();
            refreshUnorganized();
        } catch (e) {
            alert("批量整理请求失败");
        } finally {
            hideLoading();
        }
    });

    const showFilesOnlyCheckbox = document.getElementById('show-files-only');

    searchPaperBtn.addEventListener('click', async () => {
        const query = paperQueryInput.value.trim();
        if (!query) return;

        showLoading("正在检索文献...");
        try {
            const res = await fetch('/api/papers/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, n_results: 10 }) // Fetch slightly more for better deduplication
            });
            const results = await res.json();

            paperResultsContainer.innerHTML = '';
            if (results.length === 0) {
                paperResultsContainer.innerHTML = '<div class="placeholder">未找到相关内容</div>';
            } else {
                if (showFilesOnlyCheckbox.checked) {
                    // File Index Mode: Deduplicate by source
                    const uniqueFiles = [];
                    const seen = new Set();
                    results.forEach(item => {
                        if (!seen.has(item.source)) {
                            seen.add(item.source);
                            uniqueFiles.push(item);
                        }
                    });

                    uniqueFiles.forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'file-index-item';
                        div.innerHTML = `
                            <div class="file-info">
                                <i class="fas fa-file-pdf"></i>
                                <span class="filename">${item.source}</span>
                            </div>
                            <div class="file-meta">
                                <span>High Relevance Page: ${item.page}</span>
                                <span class="score">${(item.score * 100).toFixed(0)}% Match</span>
                            </div>
                        `;
                        paperResultsContainer.appendChild(div);
                    });
                } else {
                    // Standard Mode: Detailed results
                    results.slice(0, 5).forEach(item => { // Limit to 5 for standard view
                        const div = document.createElement('div');
                        div.className = 'paper-item';
                        div.innerHTML = `
                            <div class="meta">
                                <span><i class="fas fa-file-pdf"></i> ${item.source} (Page ${item.page})</span>
                                <span>Match Score: ${(item.score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="abstract-box"><strong>Abstract Match:</strong> ${item.abstract}</div>
                            <div class="snippet"><strong>Key Snippet:</strong> ${item.content.substring(0, 300)}...</div>
                        `;
                        paperResultsContainer.appendChild(div);
                    });
                }
            }
        } catch (e) {
            alert("搜索失败");
        } finally {
            hideLoading();
        }
    });

    // --- Image Management ---
    imageDropZone.addEventListener('click', () => imageInput.click());
    imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropZone.classList.add('dragover');
    });
    imageDropZone.addEventListener('dragleave', () => imageDropZone.classList.remove('dragover'));
    imageDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            imageInput.files = e.dataTransfer.files;
            imageDropZone.querySelector('p').textContent = e.dataTransfer.files[0].name;
        }
    });

    imageInput.addEventListener('change', () => {
        if (imageInput.files.length > 0) {
            imageDropZone.querySelector('p').textContent = imageInput.files[0].name;
        }
    });

    uploadImageBtn.addEventListener('click', async () => {
        if (imageInput.files.length === 0) return alert("请先选择图片");

        showLoading("正在上传并索引图片...");
        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        try {
            const res = await fetch('/api/images/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                alert("图片索引成功");
                updateStats();
            } else {
                alert(`出错: ${data.detail}`);
            }
        } catch (e) {
            alert("上传失败");
        } finally {
            hideLoading();
        }
    });

    searchImageBtn.addEventListener('click', async () => {
        const query = imageQueryInput.value.trim();
        if (!query) return;

        showLoading("正在搜索图片...");
        try {
            const res = await fetch('/api/images/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, n_results: 10 })
            });
            const results = await res.json();

            imageResultsContainer.innerHTML = '';
            if (results.length === 0) {
                imageResultsContainer.innerHTML = '<div class="placeholder">未找到匹配图片</div>';
            } else {
                results.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'img-item';

                    // Simplified: using the relative path from backend directly
                    div.innerHTML = `<img src="/${item.full_path}" alt="${item.source}">`;

                    div.onclick = () => {
                        document.querySelectorAll('.img-item').forEach(el => el.classList.remove('selected'));
                        div.classList.add('selected');
                        selectedImagePath = item.full_path;
                        visionPreview.innerHTML = `<img src="/${item.full_path}" alt="${item.source}">`;
                    };
                    imageResultsContainer.appendChild(div);
                });
            }
        } catch (e) {
            alert("搜索失败");
        } finally {
            hideLoading();
        }
    });

    // --- Vision Management ---
    describeBtn.addEventListener('click', async () => {
        if (!selectedImagePath) return alert("请先在图像管理选项卡中搜索并选择一张图片");

        showLoading("正在生成图像描述 (Florence-2)...");
        const formData = new FormData();
        formData.append('path', selectedImagePath);

        try {
            const res = await fetch('/api/vision/describe', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            visionResultText.textContent = data.description || "无描述生成";
        } catch (e) {
            alert("分析失败");
        } finally {
            hideLoading();
        }
    });

    askBtn.addEventListener('click', async () => {
        const question = visualQuestionInput.value.trim();
        if (!selectedImagePath) return alert("请先选择一张图片");
        if (!question) return alert("请输入问题");

        showLoading("正在思考答案...");
        const formData = new FormData();
        formData.append('path', selectedImagePath);
        formData.append('question', question);

        try {
            const res = await fetch('/api/vision/ask', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            visionResultText.textContent = data.answer || "未能生成答案";
        } catch (e) {
            alert("提问失败");
        } finally {
            hideLoading();
        }
    });

    // Init
    updateStats();
});
