/*
随机生成DNA数据
这是一个以机器学习为基础的简单AI程序
通过人工选择“是”与“否”，来对某一数据进行学习和选择。
最终实现随机生成虚拟角色形象。
将身高、身宽、身厚严格进行限制。
将他们的值放在一个向量中。
使用随机数不断随机，将三者随机后得来的比值结果进行限制和学习。
直到它随机出来的数据，出现在合理的区间内。
然后通过增加其他DNA数据对数据进行验证，直到确认区间值。
这个区间是可动态调整的，方便以后创作其他DNA时进行调节。

一个基于raylib的3D交互程序：随机生成长方体，点击按钮调整尺寸，三个文件会记录学习区间并自动保存。
  思路：
    1. 定义 身高 / 身宽 / 身厚 三个 int 变量，并把它们放进一个可改变的向量
    2. 每轮随机生成这三个数字，用 raylib3D 画出一个长方体模型
    3. 显示六个按钮：身高值大了 / 身高值小了 / 身宽值大了 / 身宽值小了 /身厚值大了 / 身厚值小了
    4. 点击按钮 → 程序收缩或抬高该维度的可接受区间（这就是「自然选择」）
    5. 三个维度各自的区间被分别写进三个文件：
        height.txt   width.txt   hong.txt
       完成后，把区间固定下来作为DNA基础参数 传递给其他DNA组件
*/

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// ============================================================================
//  随机数
// ============================================================================
static std::mt19937 g_rng(std::random_device{}());

static int randInt(int a, int b) {
    if (b <= a) return a;
    std::uniform_int_distribution<int> dist(a, b);
    return dist(g_rng);
}

// ============================================================================
//  中文字体
//    raylib 默认字体不含中文，这里尝试从系统常见路径加载中文字体，
//    并只预载 UI 用得到的那些字形，避免加载整个 CJK 字库。
// ============================================================================
static Font g_font = {0};

static const char* kNeededChars =
    "身高宽厚值大小了重新随机保存重置区间当前比例训练面板点击告诉"
    "AI该过或小超过范围已收敛文件导出加载默认合法增长收缩";

static std::vector<int> utf8ToCodepoints(const char* s) {
    std::vector<int> out;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s);
    while (*p) {
        int cp = 0;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p >> 5) == 0x6) {
            cp = ((*p & 0x1F) << 6) | (p[1] & 0x3F);
            p += 2;
        } else if ((*p >> 4) == 0xE) {
            cp = ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
            p += 3;
        } else {
            cp = ((*p & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
                 ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
            p += 4;
        }
        out.push_back(cp);
    }
    return out;
}

static void loadUIFont() {
    std::vector<int> cps = utf8ToCodepoints(kNeededChars);
    for (int c = 32; c < 127; ++c) cps.push_back(c);

    const char* candidates[] = {
        // Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/Deng.ttf",
        // macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        // Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        nullptr};

    for (int i = 0; candidates[i]; ++i) {
        if (FileExists(candidates[i])) {
            Font f = LoadFontEx(candidates[i], 48, cps.data(), (int)cps.size());
            if (f.texture.id != 0) {
                SetTextureFilter(f.texture, TEXTURE_FILTER_BILINEAR);
                g_font = f;
                return;
            }
        }
    }
    g_font = GetFontDefault();  // 兜底：中文会显示为方块
}

static void drawText(const std::string& s, float x, float y, float size, Color c) {
    DrawTextEx(g_font, s.c_str(), {x, y}, size, 1.0f, c);
}

static float textWidth(const std::string& s, float size) {
    return MeasureTextEx(g_font, s.c_str(), size, 1.0f).x;
}

// ============================================================================
//  维度定义
//    value : 当前随机出来的数值
//    lo/hi : 学习到的可接受区间（会随点击不断收缩）
//    def   : 出厂默认区间（重置用）
//    abs   : 物理硬边界，区间永远不会越过
// ============================================================================
struct Dimension {
    std::string name;      // 显示名：身高 / 身宽 / 身厚
    std::string filename;  // 存储文件
    int value;             // 当前值
    int lo, hi;            // 学习区间
    int defLo, defHi;      // 默认区间
    int absLo, absHi;      // 绝对边界
};

// 顺序固定为：0=身高  1=身宽  2=身厚
static std::vector<Dimension> g_dims;

// ----------------------------------------------------------------------------
// 三个 int 变量 —— 也就是题目要求的身高、身宽、身厚
// 它们被同步进 g_dims 里，并放进一个可改变的向量 g_values
// ----------------------------------------------------------------------------
static int g_height = 175;
static int g_width = 45;
static int g_thickness = 25;

// 「将他们的值放在一个可改变的向量中」
static std::vector<int*> g_values = {&g_height, &g_width, &g_thickness};

static void initDimensions() {
    g_dims.clear();
    g_dims.push_back({"身高", "height.txt", g_height,
                      120, 220, 120, 220, 100, 260});
    g_dims.push_back({"身宽", "width.txt", g_width,
                      20, 90, 20, 90, 5, 130});
    g_dims.push_back({"身厚", "thickness.txt", g_thickness,
                      10, 60, 10, 60, 3, 90});
}

// 把 g_dims 里的 value 同步回三个 int 变量（保持两者一致）
static void syncValues() {
    for (std::size_t i = 0; i < g_dims.size() && i < g_values.size(); ++i) {
        *g_values[i] = g_dims[i].value;
    }
}

// ============================================================================
//  文件读写 —— 三个文件各自保存一个维度
//  文件格式（纯文本，可手改）：
//      第一行：lo hi
// ============================================================================
static void saveDim(const Dimension& d) {
    std::ofstream f(d.filename);
    if (!f) return;
    f << d.lo << ' ' << d.hi << '\n';
}

static bool loadDim(Dimension& d) {
    std::ifstream f(d.filename);
    if (!f) return false;

    int lo = 0, hi = 0;
    if (!(f >> lo >> hi)) return false;

    // 夹进绝对边界，并保证 lo <= hi
    lo = std::max(d.absLo, std::min(lo, d.absHi));
    hi = std::max(d.absLo, std::min(hi, d.absHi));
    if (hi < lo) std::swap(lo, hi);

    d.lo = lo;
    d.hi = hi;
    return true;
}

static void saveAll() {
    for (const auto& d : g_dims) saveDim(d);
}

static void loadAll() {
    for (auto& d : g_dims) loadDim(d);
}

static void resetAll() {
    for (auto& d : g_dims) {
        d.lo = d.defLo;
        d.hi = d.defHi;
        saveDim(d);
    }
}

// ============================================================================
//  采样 / 学习
// ============================================================================

// 在三个维度各自的学习区间内随机取值
static void resample() {
    for (auto& d : g_dims) {
        d.value = randInt(d.lo, d.hi);
    }
    syncValues();
}

// ----------------------------------------------------------------------------
//  点击反馈：
//    dir = +1 表示「值大了」  → 压低上界 hi
//    dir = -1 表示「值小了」  → 抬高下界 lo
//  这就是「人工选择」的核心：用两次比较把区间一点点夹到合理范围内。
// ----------------------------------------------------------------------------
static void applyFeedback(int index, int dir) {
    if (index < 0 || index >= (int)g_dims.size()) return;

    Dimension& d = g_dims[index];
    const int v = d.value;

    if (dir > 0) {
        // 值大了：把上界压到 v-1（如果还高于下界）
        const int newHi = v - 1;
        if (newHi >= d.lo && newHi < d.hi) d.hi = newHi;
    } else {
        // 值小了：把下界抬到 v+1（如果还低于上界）
        const int newLo = v + 1;
        if (newLo <= d.hi && newLo > d.lo) d.lo = newLo;
    }

    saveDim(d);  // 每次选择都立刻落盘，避免丢失训练成果
}

// 判断所有维度是否已经收敛到足够窄的区间
static bool allConverged() {
    for (const auto& d : g_dims) {
        const int span = d.hi - d.lo;
        const int initSpan = d.defHi - d.defLo;
        if (span > std::max(4, initSpan / 6)) return false;
    }
    return true;
}

// ============================================================================
//  UI 辅助
// ============================================================================
static bool drawButton(Rectangle r, const std::string& label, float fontSize,
                       Color fill, Color fillHover, Color border,
                       Color borderHover, Color textColor) {
    const bool hover = CheckCollisionPointRec(GetMousePosition(), r);

    DrawRectangleRec(r, hover ? fillHover : fill);
    DrawRectangleLinesEx(r, 2.0f, hover ? borderHover : border);

    const float tw = textWidth(label, fontSize);
    drawText(label,
             r.x + (r.width - tw) * 0.5f,
             r.y + (r.height - fontSize) * 0.5f - 1.0f,
             fontSize, textColor);

    return hover;
}

// ============================================================================
//  main
// ============================================================================
int main() {
    const int SW = 1140;
    const int SH = 720;

    InitWindow(SW, SH, "Natural Selection - 虚拟角色训练器");
    SetTargetFPS(60);

    loadUIFont();
    initDimensions();
    loadAll();
    resample();

    // ------------------------- 3D 相机 -------------------------
    Camera3D camera = {0};
    camera.position = {22.0f, 21.0f, 22.0f};
    camera.target = {0.0f, 8.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // ------------------------- 面板布局 -------------------------
    const int PANEL_X = SW - 330;
    const int PANEL_W = 330;

    Rectangle btnBig[3], btnSmall[3];
    for (int i = 0; i < 3; ++i) {
        const int y = 80 + i * 140;
        btnBig[i] = {(float)(PANEL_X + 20), (float)(y + 60), 135.0f, 50.0f};
        btnSmall[i] = {(float)(PANEL_X + 175), (float)(y + 60), 135.0f, 50.0f};
    }

    const Rectangle btnReroll = {(float)(PANEL_X + 20), 510.0f, 290.0f, 50.0f};
    const Rectangle btnReset = {(float)(PANEL_X + 20), 575.0f, 290.0f, 50.0f};

    const float SCALE = 0.08f;  // 1 cm -> 0.08 世界单位

    // ------------------------- 主循环 -------------------------
    while (!WindowShouldClose()) {
        const Vector2 mouse = GetMousePosition();

        // ---------------- 输入处理 ----------------
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            for (int i = 0; i < 3; ++i) {
                if (CheckCollisionPointRec(mouse, btnBig[i])) {
                    applyFeedback(i, +1);
                    resample();
                } else if (CheckCollisionPointRec(mouse, btnSmall[i])) {
                    applyFeedback(i, -1);
                    resample();
                }
            }
            if (CheckCollisionPointRec(mouse, btnReroll)) {
                resample();
            }
            if (CheckCollisionPointRec(mouse, btnReset)) {
                resetAll();
                resample();
            }
        }

        // ---------------- 绘制 ----------------
        BeginDrawing();
        ClearBackground({245, 247, 250, 255});

        // ---- 3D 模型 ----
        BeginMode3D(camera);
        DrawGrid(24, 1.0f);

        const float H = g_dims[0].value * SCALE;
        const float W = g_dims[1].value * SCALE;
        const float T = g_dims[2].value * SCALE;

        const Vector3 center = {0.0f, H * 0.5f, 0.0f};

        // 地面投影，让长方体看起来是立在地上的
        DrawCube({0.0f, -0.03f, 0.0f}, W + 0.6f, 0.05f, T + 0.6f,
                 Color{205, 210, 220, 140});

        // 长方体本体
        DrawCube(center, W, H, T, Color{132, 186, 245, 255});
        DrawCubeWires(center, W, H, T, Color{38, 88, 158, 255});
        EndMode3D();

        // ---- 右侧面板背景 ----
        DrawRectangle(PANEL_X, 0, PANEL_W, SH, Color{238, 241, 246, 250});
        DrawRectangleLines(PANEL_X, 0, 1, SH, Color{198, 204, 216, 255});

        // ---- 标题 ----
        {
            const std::string title = "自然选择 · 训练面板";
            const float tw = textWidth(title, 22.0f);
            drawText(title, PANEL_X + (PANEL_W - tw) * 0.5f, 26.0f, 22.0f,
                     Color{48, 58, 80, 255});
        }

        // ---- 三个维度 ----
        for (int i = 0; i < 3; ++i) {
            const int y = 80 + i * 140;
            const auto& d = g_dims[i];

            // 当前值
            const std::string line1 =
                d.name + "：" + std::to_string(d.value) + " cm";
            drawText(line1, (float)(PANEL_X + 20), (float)y, 22.0f,
                     Color{28, 38, 60, 255});

            // 学习区间
            const std::string line2 = "学习区间 [" + std::to_string(d.lo) +
                                      ", " + std::to_string(d.hi) + "]";
            drawText(line2, (float)(PANEL_X + 20), (float)(y + 30), 16.0f,
                     Color{112, 122, 142, 255});

            // 「值大了」
            drawButton(btnBig[i], d.name + "值大了", 19.0f,
                       Color{246, 222, 216, 255}, Color{238, 182, 172, 255},
                       Color{212, 172, 166, 255}, Color{200, 88, 68, 255},
                       Color{122, 40, 30, 255});

            // 「值小了」
            drawButton(btnSmall[i], d.name + "值小了", 19.0f,
                       Color{220, 240, 223, 255}, Color{178, 216, 184, 255},
                       Color{168, 206, 172, 255}, Color{58, 148, 80, 255},
                       Color{28, 90, 46, 255});
        }

        // ---- 重新随机 ----
        drawButton(btnReroll, "重新随机", 20.0f,
                   Color{226, 233, 244, 255}, Color{208, 221, 240, 255},
                   Color{170, 182, 200, 255}, Color{88, 120, 180, 255},
                   Color{44, 58, 84, 255});

        // ---- 重置 ----
        drawButton(btnReset, "重置所有区间", 20.0f,
                   Color{245, 228, 228, 255}, Color{240, 212, 212, 255},
                   Color{210, 180, 180, 255}, Color{190, 100, 100, 255},
                   Color{130, 50, 50, 255});

        // ---- 底部提示 ----
        {
            const std::string hint = "点击按钮告诉 AI：该值大了还是小了";
            const float tw = textWidth(hint, 15.0f);
            drawText(hint, PANEL_X + (PANEL_W - tw) * 0.5f, 654.0f, 15.0f,
                     Color{140, 148, 166, 255});

            const std::string hint2 = "数据自动保存到 height/width/thickness.txt";
            const float tw2 = textWidth(hint2, 13.0f);
            drawText(hint2, PANEL_X + (PANEL_W - tw2) * 0.5f, 680.0f, 13.0f,
                     Color{162, 170, 184, 255});
        }

        // ---- 左上角状态 ----
        {
            drawText("拖动鼠标旋转视角 · 滚轮缩放", 20.0f, 20.0f, 16.0f,
                     Color{140, 148, 166, 255});

            const std::string status =
                allConverged() ? "状态：已收敛（区间足够窄）"
                               : "状态：训练中…";
            drawText(status, 20.0f, 44.0f, 16.0f,
                     allConverged() ? Color{40, 130, 70, 255}
                                    : Color{120, 128, 148, 255});
        }

        EndDrawing();
    }

    // ------------------------- 退出 -------------------------
    saveAll();

    if (g_font.texture.id != 0 &&
        g_font.texture.id != GetFontDefault().texture.id) {
        UnloadFont(g_font);
    }
    CloseWindow();
    return 0;
}
/*

训练玩法与核心机制

这个程序把“人工选择”变成了点击按钮，让模型自己慢慢学会合理的比例。

· 随机生成与3D展示：身高、身宽、身厚三个int变量被放进可改变的向量，每轮从各自区间随机取值，并实时用raylib3D画出长方体。
· 六按钮反馈：右侧面板对应每个维度的“值大了”和“值小了”，点击后立即收缩或抬高该维度的可接受区间，并自动保存到三个txt文件。
· 区间学习与收敛：每次点击都会让区间向中间夹逼，状态栏会显示“训练中”或“已收敛”，重置按钮可恢复默认区间。
· 数据持久化：三个文件分别记录每个维度的当前区间，下次启动自动读取，实现生成参数的固定与复用。
*/