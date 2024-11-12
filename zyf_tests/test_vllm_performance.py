import sys
import time
from vllm import LLM, SamplingParams

# Read parameters from command line
batch_size = int(sys.argv[1])
tensor_parallel = int(sys.argv[2])

# 128 prompts
prompts = [
    "In a hidden valley surrounded by misty mountains, a tranquil lake reflects the forest's lush green. The air is filled with the scent of pine, and sunlight gently filters through the clouds.",
    "In a bustling ancient marketplace, vendors call out their wares, colorful fabrics flutter in the breeze, and the scents of spices and fresh fruits fill the air amid lively conversations.",
    "A cozy library with towering shelves filled with books creates narrow aisles. The smell of old paper lingers in the air, and soft light filters through large, dusty windows, inviting quiet reflection.",
    "In a futuristic city, sleek skyscrapers pierce the sky, illuminated by neon lights. Flying cars zip through lanes, casting reflections on glass, as the metropolis buzzes with life.",
    "A grand ballroom with chandeliers sparkling overhead, casting light over elegantly dressed guests. The sound of a waltz fills the room as dancers glide across the gleaming marble floors.",
    "A quiet forest in autumn, where golden leaves cover the ground. A gentle breeze rustles through the trees, and sunlight filters down, casting warm dappled light across the winding path.",
    "A serene mountain lake at dawn, where the water’s surface mirrors the surrounding mountains. The first light touches the lake, casting a golden glow over everything, with soft mist in the air.",
    "In a futuristic space station orbiting a distant planet, crew members in advanced suits monitor data on holographic screens. The vastness of space stretches beyond the station's windows.",
    "A misty valley at sunrise, where fog blankets the landscape. Sunlight pierces the mist, revealing rolling hills, tall trees, and a river flowing gently through the valley below.",
    "In a dense jungle, vines hang from towering trees, and the calls of exotic birds echo through the foliage. Sunlight breaks through in patches, illuminating vibrant greens and hidden flowers.",
    "An old lighthouse on a rocky coastline, its beam sweeping across the dark ocean. The sound of waves crashes against the shore, blending with the distant calls of seabirds over the sea.",
    "A magical library filled with floating books and enchanted scrolls. The shelves stretch endlessly, and golden lights float in the air, illuminating ancient texts with an otherworldly glow.",
    "In a medieval marketplace, vendors offer fresh produce, spices, and crafted goods. The scents of freshly baked bread fill the air as villagers gather, creating a lively, bustling scene.",
    "A futuristic laboratory with advanced holographic screens, robotic arms performing tasks, and scientists in sleek white coats. The hum of machinery fills the high-tech, clinical space.",
    "A cozy mountain cabin surrounded by snow-covered trees, where smoke rises from the chimney. Inside, the fire crackles warmly, while outside, soft snowflakes drift down from the sky.",
    "In a quiet Japanese garden, cherry blossoms drift gently in the breeze. A small wooden bridge arches over a koi pond where colorful fish swim, adding tranquility to the peaceful scene.",
    "A rainy city street at night, where neon signs reflect off wet pavement, casting colorful glows. People with umbrellas pass by, and the soft sound of raindrops fills the otherwise quiet air.",
    "A grand library with towering shelves of books, the scent of old paper and leather filling the air. Soft light filters through stained glass windows, casting colorful reflections on the floor.",
    "A serene beach at sunset, where waves gently lap the shore. The sky turns pink and purple, and the gentle sound of the ocean blends with the occasional call of seagulls in the distance.",
    "In a dark forest lit by a full moon, shadows stretch between the trees. Strange sounds echo through the woods, and a narrow path winds deeper into the darkness, inviting exploration.",
    "A cozy cafe on a rainy day, where patrons sip warm drinks by foggy windows. The smell of coffee mingles with the sound of soft conversation, creating a peaceful, inviting atmosphere.",
    "In a carnival at twilight, where bright lights and vibrant colors illuminate the night. Music fills the air as people enjoy games, rides, and treats, surrounded by laughter and excitement.",
    "In a peaceful meadow filled with wildflowers, bees buzz from blossom to blossom. A gentle breeze carries the scent of flowers, and the sound of birdsong adds to the scene's tranquility.",
    "In a medieval castle courtyard, knights train with swords, banners flutter in the wind, and the clang of metal fills the air. The morning sun casts long shadows across the stone walls.",
    "A bustling cyberpunk street with neon lights flashing from towering billboards. Crowds move through narrow alleyways lined with vendors, and the hum of machinery creates a high-tech atmosphere.",
    "In a sunlit meadow, wildflowers bloom in vibrant colors as butterflies flit among the plants. The gentle hum of bees adds a peaceful rhythm to the scene, full of nature’s harmony.",
    "An underwater coral reef where colorful fish dart between coral formations. Sunlight filters down, creating patterns on the sand, and vibrant marine life moves in all directions.",
    "A futuristic city with towering skyscrapers, neon signs, and bustling streets filled with people. Flying cars zip through the air as billboards flash advertisements in bright colors.",
    "A lone castle on a misty hilltop, its turrets piercing the fog. Shadows play across the ancient stone walls as the castle looms above the quiet valley, full of untold stories and mystery.",
    "A cozy bookshop filled with rows of old books, the scent of leather and paper in the air. Soft light from vintage lamps adds a warm glow, inviting readers to explore the quiet aisles.",
    "In a carnival filled with bright lights, music, and laughter, people enjoy rides, play games, and savor the scent of cotton candy and popcorn as they wander through lively attractions.",
    "A tranquil riverside scene where willow trees lean over the water, and the river flows slowly. The surface reflects the sky, creating a peaceful, harmonious setting under a blue sky.",
    "A lone traveler on a desert path, the sun setting behind dunes. Shadows grow long as they walk, leaving footprints in the sand, with only the distant sound of the wind for company.",
    "An enchanted castle with grand tapestries, chandeliers casting warm light, and knights standing guard along the walls. A sense of history and magic fills the ancient stone halls.",
    "In a steampunk city where gears and pipes adorn buildings, and airships float in the sky. People in vintage attire walk along cobblestone streets, creating a world of mechanical wonder.",
    "A quiet mountain village covered in fresh snow, where smoke rises from chimneys, and warm light spills from windows. Children play in the snow, filling the crisp air with laughter.",
    "In a hidden cave, crystals embedded in the walls glow softly, illuminating the space with an ethereal light. Stalactites hang from the ceiling, and a small stream flows through the floor.",
    "A busy marketplace at twilight, where vendors offer spices, handmade crafts, and fresh produce. The air is filled with lively chatter, creating a vibrant atmosphere in the setting sun.",
    "In a misty forest at dawn, where fog blankets the ground, and ancient trees loom overhead. The soft glow of the sunrise filters through, casting a peaceful light on the winding path.",
    "A tranquil Japanese garden with a koi pond, carefully raked gravel, and delicate plants. The peaceful sound of a small waterfall creates a calming rhythm in the serene, ordered space.",
    "In a snowy forest where fresh snow covers the ground, a single deer stands quietly. The sound of soft crunching echoes as it moves, adding a serene touch to the winter landscape.",
    "A spaceship docks at a futuristic space station, floating in the vastness of space. Crew members in sleek suits monitor holographic displays as stars twinkle outside large viewing windows.",
    "In a neon-lit street filled with flashing signs and crowds of people. Vendors offer food, and the hum of machinery fills the air as rain drizzles, reflecting neon colors in puddles.",
    "In a magical potion shop filled with shelves of colorful bottles, strange herbs, and mysterious ingredients. The air is thick with the scent of magic, creating a sense of wonder.",
    "A grand library with shelves reaching the high ceilings, filled with ancient books. Soft light from chandeliers casts shadows, creating a quiet, reverent space full of hidden knowledge.",
    "In an underwater cave illuminated by bioluminescent fish and plants. The eerie glow reveals rock formations as the water shimmers, and the quiet adds to the otherworldly feel.",
    "A peaceful farm at sunset, where fields stretch to the horizon. The golden light of the setting sun casts long shadows as animals graze, creating a scene of calm, rural beauty.",
    "A small European village street, where cobblestone paths wind between quaint buildings. Flower boxes line the windows, and the smell of fresh bread drifts from a nearby bakery.",
    "A misty cliffside overlooking a dark ocean, with waves crashing against the rocks below. A lone figure stands on the edge, watching as the clouds roll in and obscure the distant horizon.",
    "In a magical forest with glowing mushrooms, bioluminescent plants, and strange creatures that move between the trees. A soft glow fills the air, casting an enchanting light on the scene.",
    "An ornate palace with marble columns, golden chandeliers, and rich tapestries hanging from the walls. Sunlight streams through tall windows, illuminating the opulent interior.",
    "In a bustling port, ships dock along the piers as merchants unload goods from distant lands. The smell of saltwater and spices fills the air, while sailors share tales of adventure.",
    "A futuristic cafe where robotic servers bring drinks to tables, holographic menus glow in the dim light, and people chat over coffee. The hum of technology fills the air in this busy scene.",
    "A snowy path through a dense forest, blanketed in white. The only sound is the crunch of footsteps on fresh snow as a lone figure walks, leaving a trail of footprints behind.",
    "A grand ballroom with sparkling chandeliers hanging from high ceilings, casting light on elegantly dressed guests. The sound of an orchestra fills the room as couples dance gracefully.",
    "In a bustling ancient marketplace, vendors call out their wares as colorful fabrics flutter in the breeze. The scents of spices and fresh fruits fill the air, blending with laughter and lively chatter.",
    "A serene mountaintop covered in a blanket of snow, where the sky meets the earth in shades of blue and white. A lone traveler pauses to admire the view, breathing in the crisp, cold mountain air.",
    "In a futuristic city with neon-lit skyscrapers and flying cars zipping through traffic lanes. Billboards flash bright colors, and pedestrians move along walkways suspended above bustling streets.",
    "A quiet forest in autumn, where golden leaves blanket the ground. A soft breeze rustles through the trees, and sunlight filters through branches, casting warm, dappled light across the path.",
    "In a medieval castle courtyard, knights train with swords while banners bearing family crests flutter in the wind. The clanging of metal and sounds of battle fill the crisp morning air.",
    "A peaceful lake at dawn, its surface still and mirror-like, reflecting the surrounding mountains and trees. The first light of morning touches the water, casting a soft, golden glow over everything.",
    "In an enchanted garden with flowers in impossible colors and trees bearing luminous fruit. Butterflies with iridescent wings flit among the plants, adding a magical quality to the scene.",
    "A cozy library hidden away from the world, where books line every wall from floor to ceiling. The smell of old paper fills the air, and sunlight streams through dusty windows onto quiet reading nooks.",
    "A grand ballroom with crystal chandeliers hanging from the ceiling, casting sparkling light over dancers in elegant attire. The sound of a waltz fills the room, echoing off marble walls and floors.",
    "In a dark forest under a full moon, shadows stretch long and deep. Strange sounds echo through the trees, and a narrow path winds through the underbrush, leading into the heart of the unknown.",
    "A futuristic lab filled with advanced equipment, where scientists in white coats analyze data on holographic screens. Robotic arms perform precise tasks, creating an atmosphere of high-tech innovation.",
    "A hidden waterfall deep in the jungle, surrounded by lush green foliage. The sound of rushing water fills the air as sunlight filters through the leaves, creating a serene and peaceful retreat.",
    "In a medieval village square, villagers gather around market stalls filled with fresh produce, woven baskets, and handmade crafts. Children play nearby, their laughter adding life to the bustling scene.",
    "A tranquil Japanese garden with a koi pond and a small wooden bridge arching gracefully over it. Cherry blossom petals drift in the breeze, adding a sense of peace and beauty to the surroundings.",
    "An underwater world teeming with colorful fish and coral formations. Shafts of sunlight filter down, illuminating schools of fish darting between vibrant coral and creating a scene of lively beauty.",
    "A lone lighthouse on a rocky coastline, its beam of light sweeping across the dark ocean waves. The sound of crashing waves fills the air, blending with the distant calls of seabirds in flight.",
    "In an old European street, cobblestone paths wind between historic buildings with flower boxes in the windows. The smell of fresh bread drifts from a nearby bakery, inviting passersby inside.",
    "A carnival at twilight, where bright lights and vibrant colors illuminate the night. The sound of laughter and music fills the air as people enjoy rides, games, and sweet treats under the stars.",
    "A quiet lake in autumn, surrounded by trees in shades of red, orange, and yellow. Leaves float on the water’s surface, and the gentle ripple of the lake reflects the beauty of the changing season.",
    "In a steampunk-inspired city, gears and pipes adorn buildings, and airships float above. People in vintage attire hurry along cobbled streets, surrounded by mechanical wonders and old-world charm.",
    "A misty valley at sunrise, where the fog blankets the landscape. Sunlight begins to pierce through the mist, revealing rolling hills, tall trees, and a winding river flowing through the valley.",
    "A cozy coffee shop on a rainy day, where patrons sip hot drinks by foggy windows. The soft hum of conversation and the gentle sound of rain on the glass create a warm, inviting atmosphere.",
    "In a magical forest at night, where glowing mushrooms and bioluminescent plants light up the dark. Fireflies dance in the air, adding an enchanting glow to the surroundings as silence reigns.",
    "An ornate palace with marble columns and intricate tapestries adorning the walls. Sunlight streams through large windows, illuminating the lavish furnishings and decorations in the grand hall.",
    "A stormy ocean with towering waves crashing against the rocks of a rugged coastline. The sound of thunder rumbles in the distance, and lightning briefly illuminates the dark, churning sea.",
    "In a cozy mountain cabin, a fire crackles in the stone hearth. Snow falls softly outside the window, and the scent of pine and burning wood fills the air, creating a warm, rustic atmosphere.",
    "A futuristic space station orbiting a distant planet, where astronauts monitor data on large screens. The vastness of space stretches beyond the station’s windows, filled with stars and planets.",
    "In a tranquil meadow filled with wildflowers, bees buzz from blossom to blossom. The gentle breeze carries the sweet fragrance of flowers, and the sound of birdsong adds to the peaceful scene.",
    "A bustling medieval port where ships dock along the pier. Merchants unload goods, sailors tell tales of the sea, and the smell of saltwater mingles with spices from far-off lands.",
    "A hidden cave illuminated by bioluminescent plants and crystals embedded in the walls. The eerie glow reveals strange rock formations, and an underground stream flows gently through the darkness.",
    "A quiet forest path in winter, blanketed in fresh snow. The only sound is the crunch of footsteps as a lone traveler makes their way through the trees, leaving a trail of footprints behind.",
    "A serene riverside with willow trees leaning over the water. The surface of the river reflects the blue sky, and gentle ripples flow around stones, adding a peaceful rhythm to the quiet scene.",
    "An abandoned mansion with vines creeping up the walls and broken windows. The grand staircase leads to darkened hallways, and a sense of mystery hangs heavy in the dust-filled air.",
    "A bright, colorful carnival filled with laughter and excitement. People enjoy rides, play games, and savor the smell of cotton candy and popcorn as they wander through the lively attractions.",
    "In a dense, fog-covered forest, the silhouettes of ancient trees create an eerie atmosphere. The fog dampens all sound, and the only movement is the shifting mist that cloaks the surroundings.",
    "A grand library with shelves reaching up to the high ceiling, filled with books from every era. Soft light filters through stained glass windows, casting colorful patterns on the polished floor.",
    "In a cozy cottage in the woods, smoke curls from the chimney as the scent of fresh bread wafts through the open window. Sunlight filters through the trees, creating a warm, inviting retreat.",
    "A snowy village illuminated by lanterns, where villagers bustle between houses decorated with festive lights. Children build snowmen, and laughter fills the cold, clear air of the winter night.",
    "An ancient temple hidden deep in the jungle, its stone walls covered in moss and vines. Sunlight breaks through the canopy, casting dappled light on statues and intricate carvings lost to time.",
    "A peaceful riverside village where boats drift on calm waters. The sound of laughter and gentle splashing fills the air as villagers go about their daily routines in the warm afternoon sun.",
    "In a bustling cyberpunk city, neon lights and holographic advertisements flash along the crowded streets. People navigate narrow alleys filled with shops and vendors, creating a vibrant scene.",
    "A cozy study filled with books and antiques, where a fire crackles in the hearth. A plush armchair sits near a window, inviting one to curl up with a book and escape into another world.",
    "In an enchanted castle, grand tapestries hang from the walls, and chandeliers cast a warm glow. Knights stand guard as nobles gather for a feast, filling the hall with laughter and music.",
    "A serene lake surrounded by towering pine trees, where the surface reflects the sky. A family of ducks paddles across the water, and the sound of rustling leaves adds to the peaceful ambiance.",
    "In a neon-lit city street on a rainy night, colorful reflections dance in the puddles. People with umbrellas rush by, and the hum of machinery fills the air, blending with the sound of rain.",
    "An enchanted forest filled with glowing mushrooms and luminescent plants. Strange creatures move silently between the trees, their shapes only glimpsed in the shifting, otherworldly light.",
    "A tranquil Zen garden where carefully raked sand surrounds small stone sculptures. The sound of a nearby water fountain adds a calming rhythm, creating a place of peace and reflection.",
     "In an elegant Victorian study, dark wood bookshelves line the walls, filled with leather-bound tomes. A plush armchair sits by the fireplace, casting a warm glow over the antique furniture.",
    "On a foggy seaside cliff, a lone figure stands, gazing out at the endless waves. The sound of crashing surf fills the air, and the salty sea breeze whips through their hair as they face the horizon.",
    "In an enchanted castle, grand tapestries and golden chandeliers adorn the halls. Knights' armor stands at attention along the walls, and a sense of history and magic permeates the ancient stone.",
    "In a sunlit meadow, wildflowers bloom in vibrant colors, swaying gently in the breeze. Butterflies flit from flower to flower, and the soft hum of bees adds a peaceful rhythm to the scene.",
    "In a futuristic space station orbiting a distant planet, crew members in advanced suits monitor data on holographic displays. The vastness of space stretches beyond the station's large viewing windows.",
    "In a tranquil Zen garden, carefully raked sand patterns surround meticulously placed stones. The sound of a nearby water fountain adds a calming touch, creating a haven of peace and balance.",
    "In a bustling medieval port, ships of all sizes dock along the piers. Merchants unload goods from faraway lands, while sailors share tales of adventure and danger from their voyages.",
    "In a snowy forest clearing, the ground is blanketed in untouched snow. A single deer stands silently, its breath visible in the cold air as it watches the forest with cautious curiosity.",
     "A beach at sunrise, where waves gently lap the shore. A lone seagull soars above as the first light casts a golden glow on the sands. Shells and driftwood scatter the beach, untouched by footsteps.",
    "In a medieval marketplace, vendors shout their wares as people bustle around. The scent of freshly baked bread mixes with spices, while musicians play lively tunes, filling the air with excitement.",
    "In a dense jungle, vines hang from towering trees, and the calls of exotic birds echo through the foliage. Sunlight breaks through in patches, illuminating the rich green and the occasional burst of color.",
    "On a remote island, a lighthouse stands against the crashing waves. The beam of light sweeps across the dark sea, guiding ships safely through the treacherous waters. Wind whistles through the rocks.",
    "In an underwater coral reef, vibrant fish dart between coral formations. The water is crystal clear, and beams of sunlight create shifting patterns on the sandy floor as colorful creatures dance around.",
    "In a magical forest, bioluminescent plants and glowing mushrooms light up the night. Small creatures scurry through the underbrush as fireflies flicker in the air, casting an ethereal glow on the scene.",
    "On a quiet autumn day, a lake mirrors the vibrant reds, oranges, and yellows of the surrounding trees. Leaves drift slowly on the water, and a gentle breeze carries the scent of the changing season.",
    "In a sci-fi laboratory, holographic displays and robotic arms hum with activity. Scientists in futuristic suits monitor data on screens as drones hover nearby, working on advanced technologies.",
    "A cozy mountain village with wooden cabins, smoke curling from chimneys. Snow blankets the ground, and the scent of pine fills the air. The quiet is broken only by laughter from children building snowmen.",
    "In a grand ballroom, chandeliers sparkle above as elegantly dressed guests dance to orchestral music. The marble floors gleam, and rich tapestries hang from the walls, evoking an era of opulence.",
    "In a bustling cyberpunk street, neon lights flash from towering billboards. Crowds move through narrow alleyways lined with shops, and the hum of machinery fills the air, creating a high-tech ambiance.",
    "In a hidden valley surrounded by tall misty mountains, a tranquil lake reflects the lush green of the forest. The air is crisp, filled with the scent of pine, as sunlight gently filters through clouds.",
    "In a vast desert, endless dunes stretch under a blazing sun. A lone figure treks across the sand, shadows long, as the wind shifts grains in waves. Far in the distance, mirages dance on the horizon.",
    "In an ancient forest, towering trees covered in moss create a canopy. Light filters through leaves, casting dappled shadows on the ground. The soft murmur of a stream echoes through the silence.",
    "High on a mountain peak, a small cabin stands amid snow-covered trees. Smoke rises from the chimney, and footprints mark the path leading to a view over vast valleys and jagged cliffs below.",
    "In a bustling medieval port, ships of all sizes dock along the piers. Merchants unload goods from faraway lands, while sailors share tales of adventure and danger from their voyages.",
    "In a snowy forest clearing, the ground is blanketed in untouched snow. A single deer stands silently, its breath visible in the cold air as it watches the forest with cautious curiosity.",
    "In an enchanted garden with flowers in impossible colors and trees bearing luminous fruit. Butterflies with iridescent wings flit among the plants, adding magic to the peaceful garden."
]

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=150, max_tokens=155)

# Select the appropriate prompt slice for the current batch size
batch_prompts = prompts[:batch_size]

# Initialize the LLM with the specified tensor parallel configuration
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=tensor_parallel)

# Measure latency
start_time = time.time()
outputs = llm.generate(batch_prompts, sampling_params)
end_time = time.time()
latency = (end_time - start_time) * 1000  # in milliseconds

# Calculate throughput
total_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
throughput = total_tokens / (end_time - start_time)  # tokens per second

# Print the results in the format: batch_size, tensor_parallel, latency, throughput
print(f"{batch_size} × 16\t{tensor_parallel} × RTX 3090 Ti\t{latency:.2f} ms\t{throughput:.2f} tokens/s")

# # Output inference results
# print("Inference Results:")
# for i, output in enumerate(outputs):
#     print(f"Prompt {i + 1}: {batch_prompts[i]}")
#     print(f"Generated Text: {output.outputs[0].text}\n")
