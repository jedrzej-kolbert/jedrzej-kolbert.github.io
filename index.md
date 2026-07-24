---
layout: default
title: Home
---

<section class="hero-section">
  <div class="site-container">
    <div class="hero-tag">
      <span>→</span> Empirical Alignment & Safety Research
    </div>
    <h1 class="hero-title">
      Anticipate the next frontier of model behavior.
    </h1>
    <p class="hero-subtitle">
      Investigating belief insertion, reversal cost asymmetry, and representation dynamics in modern language models.
    </p>
    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
      <a href="#research" class="nav-action-pill">
        Explore Papers <span class="arrow">↓</span>
      </a>
      <a href="https://github.com/s184361" target="_blank" rel="noopener" class="nav-action-pill" style="border-color: rgba(255,255,255,0.2);">
        GitHub Repository <span class="arrow">→</span>
      </a>
    </div>
  </div>
</section>

<section id="research" class="section-wrapper">
  <div class="site-container">
    <div class="section-header-bar">
      <div class="section-title">
        Featured Research & Publications
      </div>
      <div style="font-size: 0.85rem; color: var(--accent-gold);">
        Updated 2026 →
      </div>
    </div>

    <div class="card-grid">
      {% for post in site.posts %}
        <a href="{{ post.url | relative_url }}" class="dark-card">
          <div>
            <div class="card-top-bar">
              <span class="card-tag">RESEARCH PAPER</span>
              <span>{{ post.date | date: "%B %-d, %Y" }}</span>
            </div>
            <h2 class="card-title">{{ post.title }}</h2>
            <p class="card-description">
              {% if post.excerpt %}
                {{ post.excerpt | strip_html | truncatewords: 30 }}
              {% else %}
                An empirical study on synthetic document finetuning (SDF), false belief insertion, and reversal cost asymmetry across training schedules.
              {% endif %}
            </p>
          </div>
          <div class="card-footer">
            <span>Read Publication</span>
            <span class="arrow">→</span>
          </div>
        </a>
      {% endfor %}
    </div>
  </div>
</section>
