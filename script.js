const projectData = window.SUPPLEMENT_DATA || { meta: {}, sections: [] };

function renderPaperMeta() {
  const authors = document.getElementById("paper-authors");
  const affiliations = document.getElementById("paper-affiliations");
  const notes = document.getElementById("paper-author-notes");

  if (!authors || !affiliations || !notes) return;

  (projectData.meta.authors || []).forEach((author, index, list) => {
    const item = document.createElement("span");
    item.className = "author";

    const name = author.url ? document.createElement("a") : document.createElement("span");
    name.textContent = author.name;
    if (author.url) {
      name.href = author.url;
      name.target = "_blank";
      name.rel = "noopener noreferrer";
      name.setAttribute("aria-label", `${author.name} homepage (opens in a new tab)`);
    }
    item.appendChild(name);

    const markers = (author.affiliations || []).map(String);
    if (author.corresponding) markers.push("*");
    if (markers.length) {
      const sup = document.createElement("sup");
      sup.textContent = markers.join(",");
      item.appendChild(sup);
    }

    authors.appendChild(item);
    if (index < list.length - 1) authors.append(document.createTextNode(", "));
  });

  (projectData.meta.affiliations || []).forEach((affiliation) => {
    const item = document.createElement("span");
    const sup = document.createElement("sup");
    sup.textContent = affiliation.id;
    item.append(sup, document.createTextNode(affiliation.name));
    affiliations.appendChild(item);
  });

  if ((projectData.meta.authors || []).some((author) => author.corresponding)) {
    notes.textContent = "* Corresponding authors";
  }
}

function setupHeader() {
  const header = document.querySelector("[data-header]");
  const menuButton = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".primary-nav");
  const links = Array.from(document.querySelectorAll(".primary-nav a"));
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const onScroll = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 16);

    const marker = window.scrollY + window.innerHeight * 0.34;
    let activeId = sections[0]?.id;
    sections.forEach((section) => {
      if (section.offsetTop <= marker) activeId = section.id;
    });
    links.forEach((link) => {
      const isActive = link.getAttribute("href") === `#${activeId}`;
      link.classList.toggle("is-active", isActive);
      if (isActive) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  const closeMenu = () => {
    menuButton?.setAttribute("aria-expanded", "false");
    nav?.classList.remove("is-open");
    document.body.classList.remove("menu-open");
  };

  menuButton?.addEventListener("click", () => {
    const open = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(open));
    nav?.classList.toggle("is-open", open);
    document.body.classList.toggle("menu-open", open);
  });

  links.forEach((link) => link.addEventListener("click", closeMenu));
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

function setupReveals() {
  const items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    items.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.08, rootMargin: "0px 0px -45px" });

  items.forEach((item, index) => {
    item.style.transitionDelay = `${Math.min(index % 3, 2) * 60}ms`;
    observer.observe(item);
  });
}

function moveResultsToFront() {
  const results = document.getElementById("results");
  const hero = document.getElementById("top");
  if (results && hero) hero.after(results);
}

function setupResultCarousels() {
  const carousels = document.querySelectorAll("[data-result-carousel]");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  carousels.forEach((carousel) => {
    const viewport = carousel.querySelector(".result-viewport");
    const track = carousel.querySelector(".result-track");
    const cards = Array.from(carousel.querySelectorAll(".result-card"));
    const buttons = carousel.querySelectorAll("[data-carousel-direction]");
    let timer;

    const stepSize = () => {
      const cardWidth = cards[0]?.getBoundingClientRect().width || 0;
      const gap = Number.parseFloat(getComputedStyle(track).columnGap) || 0;
      return (cardWidth + gap) * 3;
    };

    const move = (direction) => {
      const atEnd = viewport.scrollLeft + viewport.clientWidth >= viewport.scrollWidth - 8;
      const atStart = viewport.scrollLeft <= 8;

      if (direction > 0 && atEnd) {
        viewport.scrollTo({ left: 0, behavior: "smooth" });
      } else if (direction < 0 && atStart) {
        viewport.scrollTo({ left: viewport.scrollWidth, behavior: "smooth" });
      } else {
        viewport.scrollBy({ left: stepSize() * direction, behavior: "smooth" });
      }
    };

    const stop = () => window.clearInterval(timer);
    const start = () => {
      stop();
      if (!reduceMotion) timer = window.setInterval(() => move(1), 5000);
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        move(Number(button.dataset.carouselDirection));
        start();
      });
    });

    carousel.addEventListener("mouseenter", stop);
    carousel.addEventListener("mouseleave", start);
    carousel.addEventListener("focusin", stop);
    carousel.addEventListener("focusout", start);
    viewport.addEventListener("pointerdown", stop);
    viewport.addEventListener("pointerup", start);

    cards.forEach((card) => {
      const video = card.querySelector("video");
      video.addEventListener("click", () => {
        if (video.paused) {
          video.play().catch(() => {});
          card.classList.remove("is-paused");
        } else {
          video.pause();
          card.classList.add("is-paused");
        }
      });
    });

    if ("IntersectionObserver" in window) {
      const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          const video = entry.target;
          const card = video.closest(".result-card");
          if (entry.isIntersecting && !card.classList.contains("is-paused") && !reduceMotion) {
            video.play().catch(() => {});
          } else if (!entry.isIntersecting || reduceMotion) {
            video.pause();
          }
        });
      }, { threshold: 0.35 });
      cards.forEach((card) => videoObserver.observe(card.querySelector("video")));
    }

    start();
  });
}

function makeRowControls(scope) {
  const controls = document.createElement("div");
  controls.className = "row-controls";
  controls.setAttribute("aria-label", "Video row controls");

  const play = document.createElement("button");
  play.type = "button";
  play.textContent = "Play all";

  const pause = document.createElement("button");
  pause.type = "button";
  pause.textContent = "Pause";

  play.addEventListener("click", () => {
    scope.dataset.userPaused = "false";
    scope.querySelectorAll("video").forEach((video) => video.play().catch(() => {}));
  });
  pause.addEventListener("click", () => {
    scope.dataset.userPaused = "true";
    scope.querySelectorAll("video").forEach((video) => video.pause());
  });

  controls.append(play, pause);
  return controls;
}

function makeVideoCard(method, src) {
  const card = document.createElement("figure");
  card.className = "video-card";
  card.dataset.methodId = method.id;

  const frame = document.createElement("div");
  frame.className = "video-frame";

  const video = document.createElement("video");
  video.src = src;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.setAttribute("aria-label", `${method.label} comparison video`);
  frame.appendChild(video);

  const caption = document.createElement("figcaption");
  caption.textContent = method.label;
  card.append(frame, caption);
  return card;
}

function observeComparisonRow(row) {
  if (!("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const videos = entry.target.querySelectorAll("video");
      if (entry.isIntersecting && entry.target.dataset.userPaused !== "true") {
        videos.forEach((video) => video.play().catch(() => {}));
      } else if (!entry.isIntersecting) {
        videos.forEach((video) => video.pause());
      }
    });
  }, { threshold: 0.25, rootMargin: "150px 0px" });
  observer.observe(row);
}

function renderComparison(comparison) {
  const block = document.createElement("section");
  block.className = "bit-comparison";
  block.dataset.userPaused = "false";

  const header = document.createElement("div");
  header.className = "bit-header";
  const title = document.createElement("h4");
  title.className = "bit-title";
  title.textContent = (comparison.title || "")
    .replace("4-bit", "INT4")
    .replace("3-bit", "INT3");
  header.append(title, makeRowControls(block));

  const row = document.createElement("div");
  row.className = "video-row";
  row.style.setProperty("--method-count", comparison.methods.length);
  comparison.methods.forEach((method) => {
    row.appendChild(makeVideoCard(method, comparison.videos[method.id]));
  });

  block.append(header, row);
  observeComparisonRow(block);
  return block;
}

function renderSample(sample, index) {
  const article = document.createElement("article");
  article.className = "prompt-sample";

  const header = document.createElement("div");
  header.className = "sample-header";
  const number = document.createElement("span");
  number.className = "sample-number";
  number.textContent = String(index + 1).padStart(2, "0");

  const promptBlock = document.createElement("div");
  promptBlock.className = "prompt-block";
  const label = document.createElement("span");
  label.className = "prompt-label";
  label.textContent = "Generation prompt";
  const prompt = document.createElement("p");
  prompt.className = "prompt-text";
  prompt.textContent = sample.prompt;
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "prompt-toggle";
  toggle.textContent = "Show full prompt";
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => {
    const expanded = article.classList.toggle("is-expanded");
    toggle.textContent = expanded ? "Collapse prompt" : "Show full prompt";
    toggle.setAttribute("aria-expanded", String(expanded));
  });
  promptBlock.append(label, prompt, toggle);
  header.append(number, promptBlock);

  const stack = document.createElement("div");
  stack.className = "comparison-stack";
  (sample.comparisons || []).forEach((comparison) => stack.appendChild(renderComparison(comparison)));
  article.append(header, stack);
  return article;
}

function setupExplorer() {
  const sectionTabs = document.getElementById("section-tabs");
  const modelTabs = document.getElementById("model-tabs");
  const gallery = document.getElementById("comparison-gallery");
  if (!sectionTabs || !modelTabs || !gallery || !projectData.sections?.length) return;

  let sectionIndex = 0;
  let groupIndex = 0;

  const renderSectionTabs = () => {
    sectionTabs.replaceChildren();
    projectData.sections.forEach((section, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "tab";
      button.textContent = index === 0 ? "Method comparison" : section.title;
      button.classList.toggle("is-active", index === sectionIndex);
      button.setAttribute("aria-selected", String(index === sectionIndex));
      button.addEventListener("click", () => {
        if (index === sectionIndex) return;
        sectionIndex = index;
        groupIndex = 0;
        render();
      });
      sectionTabs.appendChild(button);
    });
  };

  const renderModelTabs = () => {
    modelTabs.replaceChildren();
    const groups = projectData.sections[sectionIndex]?.groups || [];
    groups.forEach((group, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "tab";
      button.textContent = group.title;
      button.classList.toggle("is-active", index === groupIndex);
      button.setAttribute("aria-selected", String(index === groupIndex));
      button.addEventListener("click", () => {
        if (index === groupIndex) return;
        groupIndex = index;
        renderModelTabs();
        renderGallery();
      });
      modelTabs.appendChild(button);
    });
  };

  const renderGallery = () => {
    gallery.replaceChildren();
    const group = projectData.sections[sectionIndex]?.groups?.[groupIndex];
    if (!group) return;

    const head = document.createElement("div");
    head.className = "gallery-group-head";
    const title = document.createElement("h3");
    title.textContent = group.title;
    const count = document.createElement("span");
    count.textContent = `${group.samples?.length || 0} scene${group.samples?.length === 1 ? "" : "s"}`;
    head.append(title, count);

    const list = document.createElement("div");
    list.className = "sample-list";
    (group.samples || []).forEach((sample, index) => list.appendChild(renderSample(sample, index)));
    gallery.append(head, list);
  };

  const render = () => {
    renderSectionTabs();
    renderModelTabs();
    renderGallery();
  };

  render();
}

renderPaperMeta();
moveResultsToFront();
setupHeader();
setupResultCarousels();
setupExplorer();
setupReveals();
