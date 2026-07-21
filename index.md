---
layout: default
title: Home
---

Updates on my AI Safety work and projects.

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    &mdash; <em>{{ post.date | date: "%B %-d, %Y" }}</em>
  </li>
{% endfor %}
</ul>
