(() => {
    const root = document.documentElement;
    const body = document.body;
    const scrollBarInner = document.getElementById("scrollBarInner");
    const revealNodes = Array.from(document.querySelectorAll("[data-reveal]"));
    const hubButtons = Array.from(document.querySelectorAll(".network-hub"));
    const motionMedia = window.matchMedia("(prefers-reduced-motion: reduce)");

    class NetworkField {
        constructor(canvas, options = {}) {
            this.canvas = canvas;
            this.ctx = canvas ? canvas.getContext("2d") : null;
            this.reducedMotion = Boolean(options.reducedMotion);
            this.width = window.innerWidth;
            this.height = window.innerHeight;
            this.dpr = Math.min(window.devicePixelRatio || 1, 2);
            this.lastTime = 0;
            this.raf = null;
            this.nodes = [];
            this.hubAnchors = {};
            this.pulses = [];
            this.pointer = {
                x: this.width * 0.5,
                y: this.height * 0.5,
                active: false,
            };
            this.focus = null;
            this.activeHubKey = null;
            this.activeHubUntil = 0;
            this.pointerTimer = null;

            this.handleResize = this.handleResize.bind(this);
            this.handlePointerMove = this.handlePointerMove.bind(this);
            this.render = this.render.bind(this);
        }

        init() {
            if (!this.canvas || !this.ctx) {
                return;
            }

            this.resizeCanvas();
            this.seedNodes();
            this.updateHubAnchors();

            window.addEventListener("resize", this.handleResize, { passive: true });
            window.addEventListener("pointermove", this.handlePointerMove, { passive: true });

            if (this.reducedMotion) {
                this.draw(performance.now());
                return;
            }

            this.raf = window.requestAnimationFrame(this.render);
        }

        setReducedMotion(nextValue) {
            const value = Boolean(nextValue);
            if (value === this.reducedMotion) {
                return;
            }

            this.reducedMotion = value;

            if (this.reducedMotion) {
                if (this.raf) {
                    window.cancelAnimationFrame(this.raf);
                    this.raf = null;
                }
                this.draw(performance.now());
                return;
            }

            this.lastTime = 0;
            this.raf = window.requestAnimationFrame(this.render);
        }

        handleResize() {
            this.width = window.innerWidth;
            this.height = window.innerHeight;
            this.resizeCanvas();
            this.updateHubAnchors();

            const expected = this.nodeCountForViewport();
            if (Math.abs(expected - this.nodes.length) > 12) {
                this.seedNodes();
            }

            if (this.reducedMotion) {
                this.draw(performance.now());
            }
        }

        handlePointerMove(event) {
            this.pointer.x = event.clientX;
            this.pointer.y = event.clientY;
            this.pointer.active = true;

            if (this.pointerTimer) {
                window.clearTimeout(this.pointerTimer);
            }

            this.pointerTimer = window.setTimeout(() => {
                this.pointer.active = false;
            }, 1200);
        }

        resizeCanvas() {
            if (!this.canvas || !this.ctx) {
                return;
            }

            this.dpr = Math.min(window.devicePixelRatio || 1, 2);
            this.canvas.width = Math.round(this.width * this.dpr);
            this.canvas.height = Math.round(this.height * this.dpr);
            this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        }

        nodeCountForViewport() {
            const area = this.width * this.height;
            const baseline = Math.round(area / 7600);
            return Math.max(220, Math.min(360, baseline));
        }

        seedNodes() {
            const targetCount = this.nodeCountForViewport();
            this.nodes = [];

            for (let index = 0; index < targetCount; index += 1) {
                this.nodes.push(this.randomNode());
            }
        }

        randomNode() {
            const baseHue = Math.random() < 0.62 ? 192 : 155;
            return {
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * 0.52,
                vy: (Math.random() - 0.5) * 0.52,
                radius: 0.85 + Math.random() * 2.25,
                hue: baseHue + (Math.random() - 0.5) * 16,
                phase: Math.random() * Math.PI * 2,
            };
        }

        updateHubAnchors() {
            const nextAnchors = {};
            hubButtons.forEach((button) => {
                const key = button.dataset.hub;
                if (!key) {
                    return;
                }
                const rect = button.getBoundingClientRect();
                nextAnchors[key] = {
                    x: rect.left + rect.width * 0.5,
                    y: rect.top + rect.height * 0.5,
                    radius: Math.max(rect.width, rect.height) * 0.5,
                };
            });
            this.hubAnchors = nextAnchors;
        }

        jumpToHub(key) {
            const hub = this.hubAnchors[key];
            if (!hub) {
                return;
            }

            this.activeHubKey = key;
            this.activeHubUntil = performance.now() + 920;

            this.focus = {
                x: hub.x,
                y: hub.y,
                strength: 1,
                until: performance.now() + 1000,
            };

            this.pulses.push({
                x: hub.x,
                y: hub.y,
                radius: hub.radius * 0.4,
                maxRadius: Math.max(this.width, this.height) * 0.52,
                alpha: 0.95,
                speed: 5.6,
            });

            for (let index = 0; index < this.nodes.length; index += 1) {
                const node = this.nodes[index];
                const dx = node.x - hub.x;
                const dy = node.y - hub.y;
                const distance = Math.hypot(dx, dy) + 0.001;
                const power = Math.max(0, 1 - distance / 340);
                if (power <= 0) {
                    continue;
                }
                node.vx += (dx / distance) * (0.9 * power);
                node.vy += (dy / distance) * (0.9 * power);
            }
        }

        update(delta, now) {
            const drift = Math.min(1, delta / 20);
            const pointer = this.pointer.active ? this.pointer : null;
            const focus = this.focus && now < this.focus.until ? this.focus : null;

            const flowX = Math.sin(now * 0.00016) * 0.12;
            const flowY = Math.cos(now * 0.00011) * 0.1;

            for (let index = 0; index < this.nodes.length; index += 1) {
                const node = this.nodes[index];

                node.phase += 0.006 * drift;
                node.vx += Math.sin(node.phase + now * 0.00018) * 0.0022;
                node.vy += Math.cos(node.phase + now * 0.00014) * 0.0022;

                if (pointer) {
                    const dx = pointer.x - node.x;
                    const dy = pointer.y - node.y;
                    const distance = Math.hypot(dx, dy) + 1;
                    const pull = Math.max(0, 1 - distance / 280) * 0.02;
                    node.vx += (dx / distance) * pull;
                    node.vy += (dy / distance) * pull;
                }

                if (focus) {
                    const dx = focus.x - node.x;
                    const dy = focus.y - node.y;
                    const distance = Math.hypot(dx, dy) + 1;
                    const pull = Math.max(0, 1 - distance / 390) * 0.028 * focus.strength;
                    node.vx += (dx / distance) * pull;
                    node.vy += (dy / distance) * pull;
                }

                node.vx += flowX * 0.0009;
                node.vy += flowY * 0.0009;

                // Keep the field distributed so it never collapses into one dense cluster.
                const centerX = this.width * 0.5;
                const centerY = this.height * 0.5;
                const centerDx = node.x - centerX;
                const centerDy = node.y - centerY;
                const centerDistance = Math.hypot(centerDx, centerDy) + 1;
                const centerRadius = Math.min(this.width, this.height) * 0.19;
                if (centerDistance < centerRadius) {
                    const repel = (1 - centerDistance / centerRadius) * 0.01;
                    node.vx += (centerDx / centerDistance) * repel;
                    node.vy += (centerDy / centerDistance) * repel;
                }

                node.vx *= 0.985;
                node.vy *= 0.985;

                node.x += node.vx * delta * 0.06;
                node.y += node.vy * delta * 0.06;

                const margin = 28;
                if (node.x < -margin) node.x = this.width + margin;
                if (node.x > this.width + margin) node.x = -margin;
                if (node.y < -margin) node.y = this.height + margin;
                if (node.y > this.height + margin) node.y = -margin;
            }

            this.pulses = this.pulses.filter((pulse) => {
                pulse.radius += pulse.speed * drift;
                pulse.alpha *= 0.966;
                return pulse.radius < pulse.maxRadius && pulse.alpha > 0.02;
            });

            if (this.focus && now >= this.focus.until) {
                this.focus = null;
            }
        }

        draw(now) {
            if (!this.ctx) {
                return;
            }

            const context = this.ctx;
            context.clearRect(0, 0, this.width, this.height);

            const maxDistance = Math.min(300, Math.max(195, this.width * 0.24));
            const maxDistanceSquared = maxDistance * maxDistance;

            context.lineCap = "round";

            for (let index = 0; index < this.nodes.length; index += 1) {
                const nodeA = this.nodes[index];

                for (let nested = index + 1; nested < this.nodes.length; nested += 1) {
                    const nodeB = this.nodes[nested];
                    const dx = nodeA.x - nodeB.x;
                    if (Math.abs(dx) > maxDistance) {
                        continue;
                    }
                    const dy = nodeA.y - nodeB.y;
                    const distanceSquared = dx * dx + dy * dy;
                    if (distanceSquared > maxDistanceSquared) {
                        continue;
                    }

                    const distance = Math.sqrt(distanceSquared);
                    const mix = 1 - distance / maxDistance;
                    const alpha = Math.pow(mix, 1.6) * 0.62;
                    const hue = (nodeA.hue + nodeB.hue) * 0.5;

                    context.beginPath();
                    context.strokeStyle = `hsla(${hue}, 82%, 71%, ${alpha.toFixed(3)})`;
                    context.lineWidth = 0.52 + mix * 1.36;
                    context.moveTo(nodeA.x, nodeA.y);
                    context.lineTo(nodeB.x, nodeB.y);
                    context.stroke();
                }
            }

            for (let index = 0; index < this.nodes.length; index += 1) {
                const node = this.nodes[index];
                context.beginPath();
                context.fillStyle = `hsla(${node.hue.toFixed(1)}, 90%, 80%, 0.95)`;
                context.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                context.fill();
            }

            Object.entries(this.hubAnchors).forEach(([key, hub]) => {
                const isActive = key === this.activeHubKey && now < this.activeHubUntil;
                const baseRadius = isActive ? hub.radius * 1.26 : hub.radius * 0.95;
                const alpha = isActive ? 0.32 : 0.17;

                const gradient = context.createRadialGradient(
                    hub.x,
                    hub.y,
                    0,
                    hub.x,
                    hub.y,
                    baseRadius * 1.7
                );
                gradient.addColorStop(0, `rgba(125, 211, 252, ${alpha.toFixed(3)})`);
                gradient.addColorStop(1, "rgba(125, 211, 252, 0)");

                context.beginPath();
                context.fillStyle = gradient;
                context.arc(hub.x, hub.y, baseRadius * 1.7, 0, Math.PI * 2);
                context.fill();

                context.beginPath();
                context.strokeStyle = isActive
                    ? "rgba(52, 211, 153, 0.52)"
                    : "rgba(125, 211, 252, 0.22)";
                context.lineWidth = isActive ? 1.8 : 1.15;
                context.arc(hub.x, hub.y, baseRadius, 0, Math.PI * 2);
                context.stroke();
            });

            for (let index = 0; index < this.pulses.length; index += 1) {
                const pulse = this.pulses[index];
                context.beginPath();
                context.strokeStyle = `rgba(125, 211, 252, ${pulse.alpha.toFixed(3)})`;
                context.lineWidth = 1.6;
                context.arc(pulse.x, pulse.y, pulse.radius, 0, Math.PI * 2);
                context.stroke();
            }
        }

        render(now) {
            if (this.reducedMotion) {
                this.draw(now);
                return;
            }

            const previous = this.lastTime || now;
            const delta = Math.min(34, now - previous);
            this.lastTime = now;

            this.update(delta, now);
            this.draw(now);

            this.raf = window.requestAnimationFrame(this.render);
        }
    }

    const network = new NetworkField(document.getElementById("networkCanvas"), {
        reducedMotion: motionMedia.matches,
    });
    network.init();

    const updateScrollMetrics = () => {
        const scrollTop = window.pageYOffset || root.scrollTop || 0;
        const maxScroll = Math.max(1, root.scrollHeight - root.clientHeight);
        const ratio = Math.min(1, Math.max(0, scrollTop / maxScroll));

        if (scrollBarInner) {
            scrollBarInner.style.width = `${(ratio * 100).toFixed(2)}%`;
        }

        root.style.setProperty("--scroll-ratio", ratio.toFixed(4));
    };

    window.addEventListener("scroll", updateScrollMetrics, { passive: true });
    window.addEventListener("resize", () => {
        updateScrollMetrics();
        network.updateHubAnchors();
    });
    window.setTimeout(() => network.updateHubAnchors(), 80);
    updateScrollMetrics();

    const highlightTarget = (target) => {
        target.classList.add("section-focus");
        window.setTimeout(() => {
            target.classList.remove("section-focus");
        }, 950);
    };

    const setHubActiveState = (activeButton) => {
        hubButtons.forEach((button) => button.classList.remove("is-active"));
        activeButton.classList.add("is-active");

        window.setTimeout(() => {
            activeButton.classList.remove("is-active");
        }, 980);
    };

    hubButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const targetSelector = button.dataset.target;
            const hubKey = button.dataset.hub;
            const target = targetSelector ? document.querySelector(targetSelector) : null;

            if (target) {
                target.scrollIntoView({
                    behavior: motionMedia.matches ? "auto" : "smooth",
                    block: "start",
                });
                highlightTarget(target);
            }

            setHubActiveState(button);

            if (hubKey) {
                network.jumpToHub(hubKey);
            }

            body.classList.add("hub-jump");
            window.setTimeout(() => {
                body.classList.remove("hub-jump");
            }, 620);
        });
    });

    if (motionMedia.matches) {
        revealNodes.forEach((node) => node.classList.add("is-visible"));
    } else if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) {
                        return;
                    }
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                });
            },
            {
                threshold: 0.2,
                rootMargin: "0px 0px -8% 0px",
            }
        );

        revealNodes.forEach((node) => observer.observe(node));
    } else {
        revealNodes.forEach((node) => node.classList.add("is-visible"));
    }

    const accordionDetails = Array.from(document.querySelectorAll("details[data-accordion]"));
    accordionDetails.forEach((detail) => {
        detail.addEventListener("toggle", () => {
            if (!detail.open) {
                return;
            }

            const group = detail.dataset.accordion;
            accordionDetails.forEach((other) => {
                if (other === detail || other.dataset.accordion !== group) {
                    return;
                }
                other.open = false;
            });
        });
    });

    const handleMotionChange = (event) => {
        network.setReducedMotion(event.matches);
        if (event.matches) {
            revealNodes.forEach((node) => node.classList.add("is-visible"));
        }
    };

    if (typeof motionMedia.addEventListener === "function") {
        motionMedia.addEventListener("change", handleMotionChange);
    } else if (typeof motionMedia.addListener === "function") {
        motionMedia.addListener(handleMotionChange);
    }
})();
